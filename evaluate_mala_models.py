import os 
import cv2
import time
import h5py
import waterz
import mahotas
import argparse
import numpy as np
from PIL import Image
import evaluate as ev
from scipy import ndimage

def randomlabel(segmentation):
    segmentation = segmentation.astype(np.uint32)
    uid = np.unique(segmentation)
    mid = int(uid.max()) + 1
    mapping = np.zeros(mid, dtype=segmentation.dtype)
    mapping[uid] = np.random.choice(len(uid), len(uid), replace=False).astype(segmentation.dtype)#(len(uid), dtype=segmentation.dtype)
    out = mapping[segmentation]
    out[segmentation==0] = 0
    return out

def watershed(affs, seed_method, use_mahotas_watershed=True):
    affs_xy = 1.0 - 0.5*(affs[1] + affs[2])
    depth  = affs_xy.shape[0]
    fragments = np.zeros_like(affs[0]).astype(np.uint64)
    next_id = 1
    for z in range(depth):
        seeds, num_seeds = get_seeds(affs_xy[z], next_id=next_id, method=seed_method)
        if use_mahotas_watershed:
            fragments[z] = mahotas.cwatershed(affs_xy[z], seeds)
        else:
            fragments[z] = ndimage.watershed_ift((255.0*affs_xy[z]).astype(np.uint8), seeds)
        next_id += num_seeds
    return fragments

def get_seeds(boundary, method='grid', next_id=1, seed_distance=10):
    if method == 'grid':
        height = boundary.shape[0]
        width  = boundary.shape[1]
        seed_positions = np.ogrid[0:height:seed_distance, 0:width:seed_distance]
        num_seeds_y = seed_positions[0].size
        num_seeds_x = seed_positions[1].size
        num_seeds = num_seeds_x*num_seeds_y
        seeds = np.zeros_like(boundary).astype(np.int32)
        seeds[seed_positions] = np.arange(next_id, next_id + num_seeds).reshape((num_seeds_y,num_seeds_x))

    if method == 'minima':
        minima = mahotas.regmin(boundary)
        seeds, num_seeds = mahotas.label(minima)
        seeds += next_id
        seeds[seeds==next_id] = 0

    if method == 'maxima_distance':
        distance = mahotas.distance(boundary<0.5)
        maxima = mahotas.regmax(distance)
        seeds, num_seeds = mahotas.label(maxima)
        seeds += next_id
        seeds[seeds==next_id] = 0

    return seeds, num_seeds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--in_path', type=str, default='', help='path to config file')
    parser.add_argument('-gt', '--gt_path', type=str, default='../data')
    parser.add_argument('-id', '--model_id', type=int, default=1000)
    parser.add_argument('-m', '--mode', type=str, default='ac3')
    parser.add_argument('-sn', '--seg_name', type=str, default='valid_seg')
    parser.add_argument('-ss', '--start_split', type=int, default=50)  # 25, 50, 75, 100
    parser.add_argument('-es', '--end_split', type=int, default=0)  # 25, 50, 75, 100
    parser.add_argument('-mk', '--mask_fragment', type=float, default=None)
    parser.add_argument('-st', '--start_th', type=float, default=0.5)
    parser.add_argument('-et', '--end_th', type=float, default=0.5)
    parser.add_argument('-s', '--stride', type=float, default=0.1)
    args = parser.parse_args()

    affs_path = args.in_path
    gt_path = args.gt_path
    # load affs
    test_split = args.start_split - args.end_split
    affs_name = 'affs-%s-%d.hdf' % (args.mode, test_split)
    f = h5py.File(os.path.join(affs_path, affs_name), 'r')
    affs = f['main'][:]
    f.close()
    if args.mode == 'ac3':
        f = h5py.File(os.path.join(gt_path, 'snemi3d', 'AC3_labels.h5'), 'r')
        test_label = f['main'][:]
        f.close()
    elif args.mode == 'ac4':
        f = h5py.File(os.path.join(gt_path, 'snemi3d', 'AC4_labels.h5'), 'r')
        test_label = f['main'][:]
        f.close()
    elif args.mode == 'cremia':
        f = h5py.File(os.path.join(gt_path, 'cremi', 'cremiA_labels.h5'), 'r')
        test_label = f['main'][:]
        f.close()
    elif args.mode == 'cremib':
        f = h5py.File(os.path.join(gt_path, 'cremi', 'cremiB_labels.h5'), 'r')
        test_label = f['main'][:]
        f.close()
    elif args.mode == 'cremic':
        f = h5py.File(os.path.join(gt_path, 'cremi', 'cremiC_labels.h5'), 'r')
        test_label = f['main'][:]
        f.close()
    elif args.mode == 'fib':
        f = h5py.File(os.path.join(gt_path, 'fib', 'fib_labels.h5'), 'r')
        test_label = f['main'][:]
        f.close()
    else:
        raise AttributeError('No this mode!')
    if args.end_split == 0:
        test_label = test_label[-args.start_split:]
    else:
        test_label = test_label[-args.start_split:-args.end_split]

    thresholds = np.arange(args.start_th, args.end_th+args.stride, args.stride)
    thresholds = list(thresholds)

    fragments = watershed(affs, 'maxima_distance')
    if args.mask_fragment is not None:
        tt = args.mask_fragment
        print('add mask and threshold=' + str(tt))
        affs_xy = 0.5 * (affs[1] + affs[2])
        fragments[affs_xy<tt] = 0
    
    sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
    seg = waterz.agglomerate(affs,
                        thresholds,
                        gt=None,
                        fragments=fragments,
                        scoring_function=sf,
                        discretize_queue=256)
    
    f_txt = open(os.path.join(affs_path, args.seg_name+'.txt'), 'a')
    best_arand = 1000
    best_idx = 0
    split_all = []
    merge_all = []
    voi_all = []
    arand_all = []
    for idx, seg_metric in enumerate(seg):
        segmentation = seg_metric.astype(np.int32)
        segmentation, _, _ = ev.relabel_from_one(segmentation)
        voi_merge, voi_split = ev.split_vi(segmentation, test_label)
        voi_sum = voi_split + voi_merge
        arand = ev.adapted_rand_error(segmentation, test_label)
        split_all.append(voi_split)
        merge_all.append(voi_merge)
        voi_all.append(voi_sum)
        arand_all.append(arand)
        if voi_sum < best_arand:
            best_arand = voi_sum
            best_idx = idx
    print('model=%d, th=%.6f, voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
         (args.model_id, thresholds[best_idx], split_all[best_idx], merge_all[best_idx], voi_all[best_idx], arand_all[best_idx]))
    f_txt.write('model=%d, th=%.6f, voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
                (args.model_id, thresholds[best_idx], split_all[best_idx], merge_all[best_idx], voi_all[best_idx], arand_all[best_idx]))
    f_txt.write('\n')
    f_txt.close()

