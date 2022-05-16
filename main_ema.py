from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import yaml
import time
import h5py
import logging
import argparse
import numpy as np
from attrdict import AttrDict
from tensorboardX import SummaryWriter
from collections import OrderedDict
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref

import torch
import torch.nn as nn

from dataloader.data_provider_labeled import Provider
from dataloader.data_provider_unlabel_ema import Provider as Provider_unlabel
from dataloader.provider_valid import Provider_valid
from loss.loss_unlabel import MSELoss_unlabel, BCELoss_unlabel
from utils.show import show_affs, show_affs_whole
from utils.consistency_aug import convert_consistency_scale, convert_consistency_flip
from model.unet3d_mala import UNet3D_MALA
from model.model_superhuman import UNet_PNI
from utils.utils import setup_seed, execute
from utils.post_waterz import post_waterz
from utils.post_lmc import post_lmc

def sigmoid_rampup(current, rampup_length=40.0):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch, consistency=0.1, consistency_rampup=40.0):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def init_project(cfg):
    def init_logging(path):
        logging.basicConfig(
                level    = logging.INFO,
                format   = '%(message)s',
                datefmt  = '%m-%d %H:%M',
                filename = path,
                filemode = 'w')

        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        # set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    # seeds
    setup_seed(cfg.TRAIN.random_seed)
    if cfg.TRAIN.if_cuda:
        if torch.cuda.is_available() is False:
            raise AttributeError('No GPU available')

    prefix = cfg.time
    if cfg.TRAIN.resume:
        model_name = cfg.TRAIN.model_name
    else:
        model_name = prefix + '_' + cfg.NAME
    cfg.cache_path = os.path.join(cfg.TRAIN.cache_path, model_name)
    cfg.save_path = os.path.join(cfg.TRAIN.save_path, model_name)
    cfg.record_path = os.path.join(cfg.save_path, model_name)
    cfg.valid_path = os.path.join(cfg.save_path, 'valid')
    if cfg.TRAIN.resume is False:
        if not os.path.exists(cfg.cache_path):
            os.makedirs(cfg.cache_path)
        if not os.path.exists(cfg.save_path):
            os.makedirs(cfg.save_path)
        if not os.path.exists(cfg.record_path):
            os.makedirs(cfg.record_path)
        if not os.path.exists(cfg.valid_path):
            os.makedirs(cfg.valid_path)
    init_logging(os.path.join(cfg.record_path, prefix + '.log'))
    logging.info(cfg)
    writer = SummaryWriter(cfg.record_path)
    writer.add_text('cfg', str(cfg))
    return writer

def load_dataset(cfg):
    print('Caching datasets ... ', flush=True)
    t1 = time.time()
    train_provider = Provider('train', cfg)
    valid_provider = Provider_valid(cfg)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return train_provider, valid_provider

def build_model(cfg, writer, EMA=False):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')

    show_feature = False

    if cfg.MODEL.model_type == 'mala':
        print('load mala model!')
        model = UNet3D_MALA(output_nc=cfg.MODEL.output_nc, if_sigmoid=cfg.MODEL.if_sigmoid, init_mode=cfg.MODEL.init_mode_mala).to(device)
    else:
        print('load superhuman model!')
        model = UNet_PNI(in_planes=cfg.MODEL.input_nc,
                            out_planes=cfg.MODEL.output_nc,
                            filters=cfg.MODEL.filters,
                            upsample_mode=cfg.MODEL.upsample_mode,
                            decode_ratio=cfg.MODEL.decode_ratio,
                            merge_mode=cfg.MODEL.merge_mode,
                            pad_mode=cfg.MODEL.pad_mode,
                            bn_mode=cfg.MODEL.bn_mode,
                            relu_mode=cfg.MODEL.relu_mode,
                            init_mode=cfg.MODEL.init_mode,
                            show_feature=show_feature).to(device)

    if cfg.MODEL.pre_train:
        print('Load pre-trained model ...')
        ckpt_path = os.path.join('./trained_model', \
            cfg.MODEL.trained_model_name, \
            cfg.MODEL.trained_model_id+'.ckpt')
        checkpoint = torch.load(ckpt_path)
        pretrained_dict = checkpoint['model_weights']
        if cfg.MODEL.trained_gpus > 1:
            pretained_model_dict = OrderedDict()
            for k, v in pretrained_dict.items():
                name = k[7:] # remove module.
                # name = k
                pretained_model_dict[name] = v
        else:
            pretained_model_dict = pretrained_dict

        from utils.encoder_dict import ENCODER_DICT2, ENCODER_DECODER_DICT2
        model_dict = model.state_dict()
        encoder_dict = OrderedDict()
        if cfg.MODEL.if_skip == 'True':
            print('Load the parameters of encoder and decoder!')
            encoder_dict = {k: v for k, v in pretained_model_dict.items() if k.split('.')[0] in ENCODER_DECODER_DICT2}
        elif cfg.MODEL.if_skip == 'all':
            print('Load the all parameters of model!')
            encoder_dict = pretained_model_dict
        else:
            print('Load the parameters of encoder!')
            encoder_dict = {k: v for k, v in pretained_model_dict.items() if k.split('.')[0] in ENCODER_DICT2}
        model_dict.update(encoder_dict)
        model.load_state_dict(model_dict)

    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model = nn.DataParallel(model)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))

    if EMA:
        for param in model.parameters():
            param.detach_()

    return model

def resume_params(cfg, model, optimizer, resume):
    if resume:
        t1 = time.time()
        model_path = os.path.join(cfg.save_path, 'model-%06d.ckpt' % cfg.TRAIN.model_id)

        print('Resuming weights from %s ... ' % model_path, end='', flush=True)
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_weights'])
        else:
            raise AttributeError('No checkpoint found at %s' % model_path)
        print('Done (time: %.2fs)' % (time.time() - t1))
        print('valid %d' % checkpoint['current_iter'])
        return model, optimizer, checkpoint['current_iter']
    else:
        return model, optimizer, 0

def calculate_lr(iters):
    if iters < cfg.TRAIN.warmup_iters:
        current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(float(iters) / cfg.TRAIN.warmup_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
    else:
        if iters < cfg.TRAIN.decay_iters:
            current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(1 - float(iters - cfg.TRAIN.warmup_iters) / cfg.TRAIN.decay_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
        else:
            current_lr = cfg.TRAIN.end_lr
    return current_lr


def loop(cfg, train_provider, valid_provider, model, ema_model, criterion, optimizer, iters, writer):
    f_loss_txt = open(os.path.join(cfg.record_path, 'loss.txt'), 'a')
    f_valid_txt = open(os.path.join(cfg.record_path, 'valid.txt'), 'a')
    rcd_time = []
    sum_time = 0
    sum_loss = 0
    sum_labeled_loss = 0
    sum_unlabel_loss = 0
    sum_feature_loss = 0
    device = torch.device('cuda:0')
    
    if cfg.TRAIN.loss_func == 'MSELoss':
        criterion = nn.MSELoss()
    elif cfg.TRAIN.loss_func == 'BCELoss':
        criterion = nn.BCELoss()
    else:
        raise AttributeError("NO this criterion")

    if cfg.TRAIN.loss_func_unlabel == 'MSELoss':
        criterion_unlabel = MSELoss_unlabel()
    elif cfg.TRAIN.loss_func_unlabel == 'BCELoss':
        criterion_unlabel = BCELoss_unlabel()
    else:
        raise AttributeError("NO this criterion")

    train_provider_unlabel = Provider_unlabel('train', cfg)

    try:
        valid_data = cfg.DATA.valid_dataset
    except:
        valid_data = cfg.DATA.dataset_name
    if valid_data == 'snemi3d-ac3':
        valid_mode = 'ac3'
    elif valid_data == 'snemi3d-ac4':
        valid_mode = 'ac4'
    elif valid_data == 'cremi-C':
        valid_mode = 'cremic'
    elif valid_data == 'cremi-A':
        valid_mode = 'cremia'
    elif valid_data == 'cremi-B':
        valid_mode = 'cremib'
    elif valid_data == 'fib-25':
        valid_mode = 'fib'
    elif valid_data == 'cremi-all':
        valid_mode = 'cremic'
    else:
        raise NotImplementedError
    start_split = cfg.DATA.test_split
    end_split = 0
    test_split = start_split - end_split
    seg_name = 'waterz_' + valid_mode + '_' + str(test_split) + '.txt'
    f_seg_txt = open(os.path.join(cfg.record_path, seg_name), 'a')

    while iters <= cfg.TRAIN.total_iters:
        # train
        model.train()
        ema_model.train()
        iters += 1
        t1 = time.time()
        inputs, target, _ = train_provider.next()
        inputs_unlabel_aug, inputs_unlabel_gt, det_sizes, rules = train_provider_unlabel.next()

        # decay learning rate
        if cfg.TRAIN.end_lr == cfg.TRAIN.base_lr:
            current_lr = cfg.TRAIN.base_lr
        else:
            current_lr = calculate_lr(iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        optimizer.zero_grad()
        # cat labeled and unlabeled data
        batchsize = inputs.shape[0]
        inputs_all = torch.cat([inputs, inputs_unlabel_aug], dim=0)
        pred_all = model(inputs_all)
        pred = pred_all[0:batchsize, ...]
        pred_unlabel = pred_all[batchsize:, ...]

        with torch.no_grad():
            pred_unlabel_gt = ema_model(inputs_unlabel_gt)

        # convert flip
        if cfg.DATA.if_filp_aug_unlabel:
            pred_unlabel_gt = convert_consistency_flip(pred_unlabel_gt, rules)
        # gen pseudo label and mask
        if cfg.DATA.if_scale_aug_unlabel:
            pred_unlabel_gt, masks = convert_consistency_scale(pred_unlabel_gt, det_sizes)
        else:
            masks = torch.ones_like(pred_unlabel_gt)

        ##############################
        # LOSS
        loss_labeled = criterion(pred, target)
        if cfg.TRAIN.weight_unlabel_fixed:
            loss_unlabel = cfg.TRAIN.weight_unlabel * criterion_unlabel(pred_unlabel, pred_unlabel_gt, masks)
        else:
            max_iterations = 100000
            consistency_weight = get_current_consistency_weight(iters, consistency=cfg.TRAIN.weight_unlabel*0.1, consistency_rampup=max_iterations)
            loss_unlabel = consistency_weight * criterion_unlabel(pred_unlabel, pred_unlabel_gt, masks)

        loss = loss_labeled + loss_unlabel
        loss.backward()
        ##############################

        if cfg.TRAIN.weight_decay is not None:
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.data = param.data.add(-cfg.TRAIN.weight_decay * group['lr'], param.data)
        optimizer.step()
        ema_decay = cfg.TRAIN.ema_decay   # default=0.999
        update_ema_variables(model, ema_model, ema_decay, iters)

        sum_loss += loss.item()
        sum_labeled_loss += loss_labeled.item()
        sum_unlabel_loss += loss_unlabel.item()
        sum_time += time.time() - t1

        # log train
        if iters % cfg.TRAIN.display_freq == 0 or iters == 1:
            rcd_time.append(sum_time)
            if iters == 1:
                logging.info('step %d, loss = %.6f, labeled_loss=%.6f, unlabel_loss=%.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                            % (iters, sum_loss, sum_labeled_loss, sum_unlabel_loss, current_lr, sum_time,
                            (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                writer.add_scalar('loss', sum_loss * 1, iters)
            else:
                logging.info('step %d, loss = %.6f, labeled_loss=%.6f, unlabel_loss=%.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)' \
                            % (iters, sum_loss / cfg.TRAIN.display_freq * 1, \
                            sum_labeled_loss / cfg.TRAIN.display_freq * 1, \
                            sum_unlabel_loss / cfg.TRAIN.display_freq * 1, current_lr, sum_time, \
                            (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                writer.add_scalar('loss', sum_loss / cfg.TRAIN.display_freq * 1, iters)
            f_loss_txt.write('step = %d, loss = %.6f, labeled_loss=%.6f, unlabel_loss=%.6f' \
                            % (iters, sum_loss / cfg.TRAIN.display_freq * 1, \
                            sum_labeled_loss / cfg.TRAIN.display_freq * 1, \
                            sum_unlabel_loss / cfg.TRAIN.display_freq * 1))
            f_loss_txt.write('\n')
            f_loss_txt.flush()
            sys.stdout.flush()
            sum_time = 0
            sum_loss = 0
            sum_labeled_loss = 0
            sum_unlabel_loss = 0

        # display
        if iters % cfg.TRAIN.valid_freq == 0 or iters == 1:
            # show_affs(iters, inputs, pred, target, cfg.cache_path, model_type=cfg.MODEL.model_type)
            show_affs(iters, inputs_unlabel_aug, pred_unlabel, pred_unlabel_gt, cfg.cache_path, model_type=cfg.MODEL.model_type)

        # valid
        if cfg.TRAIN.if_valid:
            if iters % cfg.TRAIN.save_freq == 0 and iters >= cfg.TRAIN.min_valid_iters:
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                model.eval()
                ema_model.eval()
                dataloader = torch.utils.data.DataLoader(valid_provider, batch_size=1, num_workers=0,
                                                shuffle=False, drop_last=False, pin_memory=True)
                losses_valid = []
                for k, batch in enumerate(dataloader, 0):
                    inputs, target, _ = batch
                    inputs = inputs.cuda()
                    target = target.cuda()
                    with torch.no_grad():
                        # pred = model(inputs)
                        pred = ema_model(inputs)
                    tmp_loss = criterion(pred, target)
                    losses_valid.append(tmp_loss.item())
                    valid_provider.add_vol(np.squeeze(pred.data.cpu().numpy()))
                epoch_loss = sum(losses_valid) / len(losses_valid)
                out_affs = valid_provider.get_results()
                gt_affs = valid_provider.get_gt_affs()
                gt_seg = valid_provider.get_gt_lb()
                valid_provider.reset_output()
                show_affs_whole(iters, out_affs, gt_affs, cfg.valid_path)

                # f_affs = h5py.File(os.path.join(cfg.record_path, 'affs-%s-%d.hdf' % (valid_mode, test_split)), 'w')
                # f_affs.create_dataset('main', data=out_affs, dtype=np.float32, compression='gzip')
                # f_affs.close()

                # for post-processing
                # for python3
                try:
                    pred_seg = post_waterz(out_affs)
                    # pred_seg = post_lmc(out_affs)
                    arand = adapted_rand_ref(gt_seg, pred_seg, ignore_labels=(0))[0]
                    voi_split, voi_merge = voi_ref(gt_seg, pred_seg, ignore_labels=(0))
                    voi_sum = voi_split + voi_merge
                except:
                    print('model-%d, segmentation failed!' % iters)
                    arand = 0.0
                    voi_split = 0.0
                    voi_merge = 0.0
                    voi_sum = 0.0

                # MSE
                whole_mse = np.sum(np.square(out_affs - gt_affs)) / np.size(gt_affs)
                out_affs = np.clip(out_affs, 0.000001, 0.999999)
                bce = -(gt_affs * np.log(out_affs) + (1 - gt_affs) * np.log(1 - out_affs))
                whole_bce = np.sum(bce) / np.size(gt_affs)
                print('model-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f' % \
                    (iters, epoch_loss, whole_mse, whole_bce), flush=True)
                writer.add_scalar('valid/epoch_loss', epoch_loss, iters)
                writer.add_scalar('valid/mse_loss', whole_mse, iters)
                writer.add_scalar('valid/bce_loss', whole_bce, iters)
                f_valid_txt.write('model-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f' % \
                                (iters, epoch_loss, whole_mse, whole_bce))
                f_valid_txt.write('\n')
                f_valid_txt.flush()

                print('model-%d, voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
                    (iters, voi_split, voi_merge, voi_sum, arand))
                f_seg_txt.write('model=%d, voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
                            (iters, voi_split, voi_merge, voi_sum, arand))
                f_seg_txt.write('\n')
                f_seg_txt.flush()
                torch.cuda.empty_cache()

                # for post-processing
                # for python2
                # try:
                #     seg_name1 = 'waterz_' + valid_mode + '_' + str(test_split)
                #     cmd_val = ['python2','evaluate_mala_models2.py', '-in',cfg.record_path, '-gt',cfg.DATA.data_folder, 
                #                 '-id',str(iters), '-m',valid_mode, '-sn',seg_name1, '-ss',str(start_split),'-es',str(end_split)]
                #     for path in execute(cmd_val):
                #         print(path, end="")
                # except:
                #     print('model-%d, segmentation failed!' % iters)

                # try:
                #     seg_name2 = 'lmc_' + valid_mode + '_' + str(test_split)
                #     cmd_val = ['python','evaluate_lmc_models.py', '-in',cfg.record_path, '-gt',cfg.DATA.data_folder, 
                #                 '-id',str(iters), '-m',valid_mode, '-sn',seg_name2, '-ss',str(start_split),'-es',str(end_split)]
                #     for path in execute(cmd_val):
                #         print(path, end="")
                # except:
                #     print('model-%d, segmentation failed!' % iters)

        # save
        if iters % cfg.TRAIN.save_freq == 0:
            states = {'current_iter': iters, 'valid_result': None,
                    'model_weights': ema_model.state_dict()}
            torch.save(states, os.path.join(cfg.save_path, 'model-%06d.ckpt' % iters))
            print('***************save modol, iters = %d.***************' % (iters), flush=True)
    f_loss_txt.close()
    f_valid_txt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='seg_inpainting', help='path to config file')
    parser.add_argument('-m', '--mode', type=str, default='train', help='path to config file')
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    print('mode: ' + args.mode)

    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f))

    timeArray = time.localtime()
    time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S', timeArray)
    print('time stamp:', time_stamp)

    cfg.path = cfg_file
    cfg.time = time_stamp

    if args.mode == 'train':
        writer = init_project(cfg)
        train_provider, valid_provider = load_dataset(cfg)
        model = build_model(cfg, writer)
        ema_model = build_model(cfg, writer, EMA=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999),
                                    eps=0.01, weight_decay=1e-6, amsgrad=True)
        model, optimizer, init_iters = resume_params(cfg, model, optimizer, cfg.TRAIN.resume)
        loop(cfg, train_provider, valid_provider, model, ema_model, nn.L1Loss(), optimizer, init_iters, writer)
        writer.close()
    else:
        pass
    print('***Done***')