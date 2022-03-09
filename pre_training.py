from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import yaml
import time
import cv2
import random
import logging
import argparse
import numpy as np
from PIL import Image
from attrdict import AttrDict
from tensorboardX import SummaryWriter
from collections import OrderedDict
import multiprocessing as mp
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dataloader.data_provider_pretraining import Provider
from dataloader.provider_valid_pretraining import Provider_valid
from loss.loss import BCELoss, WeightedBCELoss, MSELoss
from utils.show import show_affs, show_affs_whole
from model.unet3d_mala import UNet3D_MALA
from model.model_superhuman import UNet_PNI, UNet_PNI_Noskip, UNet_PNI_Noskip2
from utils.utils import setup_seed


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
    # cfg.record_path = os.path.join(cfg.TRAIN.record_path, 'log')
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
    print('Caching datasets ... ', end='', flush=True)
    t1 = time.time()
    train_provider = Provider('train', cfg)
    valid_provider = Provider_valid(cfg)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return train_provider, valid_provider

def build_model(cfg, writer):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')
    if cfg.MODEL.model_type == 'mala':
        print('load mala model!')
        model = UNet3D_MALA(output_nc=cfg.MODEL.output_nc, if_sigmoid=cfg.MODEL.if_sigmoid, init_mode=cfg.MODEL.init_mode_mala).to(device)
    else:
        if cfg.MODEL.if_skip == 'True':
            print('load superhuman model with skip!')
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
                            if_sigmoid=cfg.MODEL.if_sigmoid).to(device)
        elif cfg.MODEL.if_skip == 'False':
            print('load superhuman model without skip!')
            model = UNet_PNI_Noskip(in_planes=cfg.MODEL.input_nc,
                                    out_planes=cfg.MODEL.output_nc,
                                    filters=cfg.MODEL.filters,
                                    upsample_mode=cfg.MODEL.upsample_mode,
                                    decode_ratio=cfg.MODEL.decode_ratio,
                                    merge_mode=cfg.MODEL.merge_mode,
                                    pad_mode=cfg.MODEL.pad_mode,
                                    bn_mode=cfg.MODEL.bn_mode,
                                    relu_mode=cfg.MODEL.relu_mode,
                                    init_mode=cfg.MODEL.init_mode,
                                    if_sigmoid=cfg.MODEL.if_sigmoid).to(device)
        elif cfg.MODEL.if_skip == 'False2':
            print('load superhuman model without skip2!')
            model = UNet_PNI_Noskip2(in_planes=cfg.MODEL.input_nc,
                                    out_planes=cfg.MODEL.output_nc,
                                    filters=cfg.MODEL.filters,
                                    upsample_mode=cfg.MODEL.upsample_mode,
                                    decode_ratio=cfg.MODEL.decode_ratio,
                                    merge_mode=cfg.MODEL.merge_mode,
                                    pad_mode=cfg.MODEL.pad_mode,
                                    bn_mode=cfg.MODEL.bn_mode,
                                    relu_mode=cfg.MODEL.relu_mode,
                                    init_mode=cfg.MODEL.init_mode,
                                    if_sigmoid=cfg.MODEL.if_sigmoid).to(device)
        else:
            raise AttributeError('No this skip mode!')

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
    return model

def resume_params(cfg, model, optimizer, resume):
    if resume:
        t1 = time.time()
        model_path = os.path.join(cfg.save_path, 'model-%06d.ckpt' % cfg.TRAIN.model_id)

        print('Resuming weights from %s ... ' % model_path, end='', flush=True)
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_weights'])
            # optimizer.load_state_dict(checkpoint['optimizer_weights'])
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


def loop(cfg, train_provider, valid_provider, model, criterion, optimizer, iters, writer):
    f_loss_txt = open(os.path.join(cfg.record_path, 'loss.txt'), 'a')
    f_valid_txt = open(os.path.join(cfg.record_path, 'valid.txt'), 'a')
    rcd_time = []
    sum_time = 0
    sum_loss = 0
    device = torch.device('cuda:0')
    
    if cfg.TRAIN.loss_func == 'MSE':
        print('L2 loss...')
        criterion = nn.MSELoss()
    elif cfg.TRAIN.loss_func == 'L1':
        print('L1 loss...')
        criterion = nn.L1Loss()
    else:
        raise AttributeError("NO this criterion")

    while iters <= cfg.TRAIN.total_iters:
        # train
        model.train()
        iters += 1
        t1 = time.time()
        inputs, target, _ = train_provider.next()
        
        # decay learning rate
        if cfg.TRAIN.end_lr == cfg.TRAIN.base_lr:
            current_lr = cfg.TRAIN.base_lr
        else:
            current_lr = calculate_lr(iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        
        optimizer.zero_grad()
        pred = model(inputs)

        ##############################
        # LOSS
        loss = criterion(pred, target)
        loss.backward()
        ##############################

        if cfg.TRAIN.weight_decay is not None:
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.data = param.data.add(-cfg.TRAIN.weight_decay * group['lr'], param.data)
        optimizer.step()
        
        sum_loss += loss.item()
        sum_time += time.time() - t1
        
        # log train
        if iters % cfg.TRAIN.display_freq == 0 or iters == 1:
            rcd_time.append(sum_time)
            if iters == 1:
                logging.info('step %d, loss = %.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                            % (iters, sum_loss * 1, current_lr, sum_time,
                            (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                writer.add_scalar('loss', sum_loss * 1, iters)
            else:
                logging.info('step %d, loss = %.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                            % (iters, sum_loss / cfg.TRAIN.display_freq * 1, current_lr, sum_time,
                            (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                writer.add_scalar('loss', sum_loss / cfg.TRAIN.display_freq * 1, iters)
            f_loss_txt.write('step = ' + str(iters) + ', loss = ' + str(sum_loss / cfg.TRAIN.display_freq * 1))
            f_loss_txt.write('\n')
            f_loss_txt.flush()
            sys.stdout.flush()
            sum_time = 0
            sum_loss = 0
        
        # display
        if iters % cfg.TRAIN.valid_freq == 0 or iters == 1:
            pred = torch.cat([pred, pred, pred], dim=1)
            target = torch.cat([target, target, target], dim=1)
            show_affs(iters, inputs, pred, target, cfg.cache_path, model_type=cfg.MODEL.model_type)
        
        # valid
        if iters % cfg.TRAIN.save_freq == 0 or iters == 1:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            model.eval()
            dataloader = torch.utils.data.DataLoader(valid_provider, batch_size=1, num_workers=0,
                                             shuffle=False, drop_last=False, pin_memory=True)
            losses_valid = []
            for k, batch in enumerate(dataloader, 0):
                inputs, target, _ = batch
                inputs = inputs.cuda()
                target = target.cuda()
                with torch.no_grad():
                    pred = model(inputs)
                tmp_loss = criterion(pred, target)
                losses_valid.append(tmp_loss.item())
                valid_provider.add_vol(np.squeeze(pred.data.cpu().numpy(), axis=0))
            epoch_loss = sum(losses_valid) / len(losses_valid)
            out_affs = valid_provider.get_results()
            gt_affs = valid_provider.get_gt_affs().copy()
            valid_provider.reset_output()

            # MSE
            whole_mse = np.sum(np.square(out_affs - gt_affs)) / np.size(gt_affs)
            whole_bce = 0.0
            whole_arand = 0.0

            out_affs_show = np.repeat(out_affs, 3, 0)
            gt_affs_show = np.repeat(gt_affs, 3, 0)
            show_affs_whole(iters, out_affs_show, gt_affs_show, cfg.valid_path)

            # out_affs = np.clip(out_affs, 0.000001, 0.999999)
            # bce = -(gt_affs * np.log(out_affs) + (1 - gt_affs) * np.log(1 - out_affs))
            # whole_bce = np.sum(bce) / np.size(gt_affs)
            # out_affs[out_affs <= 0.5] = 0
            # out_affs[out_affs > 0.5] = 1
            # whole_arand = 1 - f1_score(gt_affs.astype(np.uint8).flatten(), out_affs.astype(np.uint8).flatten())
            print('model-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, ARAND-loss=%.6f' % \
                (iters, epoch_loss, whole_mse, whole_bce, whole_arand), flush=True)
            writer.add_scalar('valid/epoch_loss', epoch_loss, iters)
            writer.add_scalar('valid/mse_loss', whole_mse, iters)
            writer.add_scalar('valid/bce_loss', whole_bce, iters)
            writer.add_scalar('valid/arand_loss', whole_arand, iters)
            f_valid_txt.write('model-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, ARAND-loss=%.6f' % \
                            (iters, epoch_loss, whole_mse, whole_bce, whole_arand))
            f_valid_txt.write('\n')
            f_valid_txt.flush()
            torch.cuda.empty_cache()

        # save
        if iters % cfg.TRAIN.save_freq == 0:
            states = {'current_iter': iters, 'valid_result': None,
                    'model_weights': model.state_dict()}
            torch.save(states, os.path.join(cfg.save_path, 'model-%06d.ckpt' % iters))
            print('***************save modol, iters = %d.***************' % (iters), flush=True)
    f_loss_txt.close()
    f_valid_txt.close()


if __name__ == "__main__":
    # mp.set_start_method('spawn')
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
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999),
                                 eps=0.01, weight_decay=1e-6, amsgrad=True)
        # optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
        # optimizer = optim.Adamax(model.parameters(), lr=cfg.TRAIN.base_l, eps=1e-8)
        model, optimizer, init_iters = resume_params(cfg, model, optimizer, cfg.TRAIN.resume)
        loop(cfg, train_provider, valid_provider, model, nn.L1Loss(), optimizer, init_iters, writer)
        writer.close()
    else:
        pass
    print('***Done***')