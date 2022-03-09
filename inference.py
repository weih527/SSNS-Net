import os
import cv2
import h5py
import yaml
import torch
import argparse
import numpy as np
from skimage import morphology
from attrdict import AttrDict
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

from dataloader.provider_valid import Provider_valid
from loss.loss import BCELoss, WeightedBCELoss, MSELoss
from model.unet3d_mala import UNet3D_MALA
from model.model_superhuman import UNet_PNI
# from utils.malis_loss import malis_loss

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='seg_snemi3d_d5_1024_u200', help='path to config file')
    parser.add_argument('-mn', '--model_name', type=str, default=None)
    parser.add_argument('-id', '--model_id', type=str, default=None)
    parser.add_argument('-m', '--mode', type=str, default='snemi3d-ac3')
    parser.add_argument('-ts', '--test_split', type=int, default=50)
    parser.add_argument('-nz', '--num_z', type=int, default=18)
    parser.add_argument('-s', '--save', action='store_false', default=True)
    parser.add_argument('-sw', '--show', action='store_true', default=False)
    parser.add_argument('-malis', '--malis_loss', action='store_true', default=False)
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)

    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f))

    if args.model_name is not None:
        trained_model = args.model_name
    else:
        trained_model = cfg.TEST.model_name
    out_path = os.path.join('./inference', trained_model, args.mode)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    img_folder = 'affs_'+args.model_id
    out_affs = os.path.join(out_path, img_folder)
    if not os.path.exists(out_affs):
        os.makedirs(out_affs)
    print('out_path: ' + out_affs)
    affs_img_path = os.path.join(out_affs, 'affs_img')
    if not os.path.exists(affs_img_path):
        os.makedirs(affs_img_path)

    device = torch.device('cuda:0')
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
                        init_mode=cfg.MODEL.init_mode).to(device)

    ckpt_path = os.path.join('./trained_model', trained_model, args.model_id+'.ckpt')
    checkpoint = torch.load(ckpt_path)

    new_state_dict = OrderedDict()
    state_dict = checkpoint['model_weights']
    for k, v in state_dict.items():
        # name = k[7:] # remove module.
        name = k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)

    valid_provider = Provider_valid(cfg, valid_data=args.mode, num_z=args.num_z, test_split=args.test_split)
    val_loader = torch.utils.data.DataLoader(valid_provider, batch_size=1)

    if cfg.TRAIN.loss_func == 'MSE':
        criterion = MSELoss()
    elif cfg.TRAIN.loss_func == 'WeightedBCELoss':
        criterion = WeightedBCELoss()
    elif cfg.TRAIN.loss_func == 'BCELoss':
        criterion = BCELoss()
    else:
        raise AttributeError("NO this criterion")

    model.eval()
    loss_all = []
    f_txt = open(os.path.join(out_affs, 'scores.txt'), 'w')
    print('the number of sub-volume:', len(valid_provider))
    losses_valid = []
    pbar = tqdm(total=len(valid_provider))
    for k, data in enumerate(val_loader, 0):
        inputs, target, weightmap = data
        inputs = inputs.cuda()
        target = target.cuda()
        weightmap = weightmap.cuda()
        with torch.no_grad():
            pred = model(inputs)
            # pred, _ = model(inputs, turnoff_drop=True)
            pred = pred[:, :3, ...]
        tmp_loss = criterion(pred, target, weightmap)
        losses_valid.append(tmp_loss.item())
        valid_provider.add_vol(np.squeeze(pred.data.cpu().numpy()))
        pbar.update(1)
    pbar.close()
    epoch_loss = sum(losses_valid) / len(losses_valid)
    output_affs = valid_provider.get_results()
    gt_affs = valid_provider.get_gt_affs()
    gt_seg = valid_provider.get_gt_lb()
    valid_provider.reset_output()

    # save
    if args.save:
        print('save affs...')
        f = h5py.File(os.path.join(out_affs, 'affs.hdf'), 'w')
        f.create_dataset('main', data=output_affs, dtype=np.float32, compression='gzip')
        f.close()

    # compute MSE
    print('MSE...')
    output_affs_prop = output_affs.copy()
    whole_mse = np.sum(np.square(output_affs - gt_affs)) / np.size(gt_affs)
    print('BCE...')
    output_affs = np.clip(output_affs, 0.000001, 0.999999)
    bce = -(gt_affs * np.log(output_affs) + (1 - gt_affs) * np.log(1 - output_affs))
    whole_bce = np.sum(bce) / np.size(gt_affs)
    output_affs[output_affs <= 0.5] = 0
    output_affs[output_affs > 0.5] = 1
    print('ARAND...')
    # whole_arand = 1 - f1_score(gt_affs.astype(np.uint8).flatten(), output_affs.astype(np.uint8).flatten())
    whole_arand = 0.0
    # new
    print('F1...')
    # whole_arand_bound = f1_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - output_affs.astype(np.uint8).flatten())
    whole_arand_bound = 0.0
    print('mAP...')
    # whole_map = average_precision_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - output_affs_prop.flatten())
    whole_map = 0.0
    print('AUC...')
    # whole_auc = roc_auc_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - output_affs_prop.flatten())
    whole_auc = 0.0
    ###################################################
    if args.malis_loss:
        # from utils.malis_loss import malis_loss
        # print('Malis...')
        # t1 = time.time()
        # try:
        #     malis = malis_loss(output_affs_prop, gt_affs, gt_seg)
        # except:
        #     malis = 0.0
        # print('COST TIME: ' + str(time.time() - t1))
        malis = 0.0
    else:
        malis = 0.0
    ###################################################
    print('model-%s, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, ARAND-loss=%.6f, F1-bound=%.6f, mAP=%.6f, auc=%.6f, malis-loss=%.6f' % \
        (args.model_id, epoch_loss, whole_mse, whole_bce, whole_arand, whole_arand_bound, whole_map, whole_auc, malis))
    f_txt.write('model-%s, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, ARAND-loss=%.6f, F1-bound=%.6f, mAP=%.6f, auc=%.6f, malis-loss=%.6f' % \
                (args.model_id, epoch_loss, whole_mse, whole_bce, whole_arand, whole_arand_bound, whole_map, whole_auc, malis))
    f_txt.close()

    # show
    if args.show:
        print('show affs...')
        output_affs_prop = (output_affs_prop * 255).astype(np.uint8)
        output_affs_prop = np.transpose(output_affs_prop, (1,2,3,0))
        gt_affs = (gt_affs * 255).astype(np.uint8)
        gt_affs = np.transpose(gt_affs, (1,2,3,0))
        for i in range(output_affs_prop.shape[0]):
            img = output_affs_prop[i]
            lb = gt_affs[i]
            im_cat = np.concatenate([img, lb], axis=1)
            cv2.imwrite(os.path.join(affs_img_path, str(i).zfill(4)+'.png'), im_cat)
    print('Done')

