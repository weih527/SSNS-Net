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
    parser.add_argument('-mn', '--model_name', type=str, default='None')
    parser.add_argument('-id', '--model_id', type=str, default=None)
    parser.add_argument('-m', '--mode', type=str, default='snemi3d-ac3')
    parser.add_argument('-ts', '--test_split', type=int, default=50)
    parser.add_argument('-nz', '--num_z', type=int, default=18)
    parser.add_argument('-mk', '--mask_fragment', type=float, default=None)
    parser.add_argument('-sw', '--show', action='store_true', default=False)
    parser.add_argument('-st', '--start_th', type=float, default=0.5)
    parser.add_argument('-et', '--end_th', type=float, default=0.5)
    parser.add_argument('-s', '--stride', type=float, default=0.1)
    args = parser.parse_args()
    
    trained_model = args.model_name
    out_path = os.path.join('./inference', trained_model, args.mode)
    img_folder = 'affs_'+args.model_id
    out_affs = os.path.join(out_path, img_folder)
    print('out_path: ' + out_affs)
    seg_img_path = os.path.join(out_affs, 'seg_waterz')
    if not os.path.exists(seg_img_path):
        os.makedirs(seg_img_path)

    # load affs
    f = h5py.File(os.path.join(out_affs, 'affs.hdf'), 'r')
    affs = f['main'][:]
    f.close()

    if args.mode == 'snemi3d-ac3' or args.mode == 'snemi3d':
        data_path = './data/snemi3d'
        f_raw = h5py.File(os.path.join(data_path, 'AC3_inputs.h5'), 'r')
        raw = f_raw['main'][:]
        f_raw.close()
        raw = raw[-50:]

        f_label = h5py.File(os.path.join(data_path, 'AC3_labels.h5'), 'r')
        gt = f_label['main'][:]
        f_label.close()
        gt = gt[-50:]
    elif args.mode == 'snemi3d-ac4':
        data_path = './data/snemi3d'
        f_raw = h5py.File(os.path.join(data_path, 'AC4_inputs.h5'), 'r')
        raw = f_raw['main'][:]
        f_raw.close()
        raw = raw[-50:]

        f_label = h5py.File(os.path.join(data_path, 'AC4_labels.h5'), 'r')
        gt = f_label['main'][:]
        f_label.close()
        gt = gt[-50:]
    elif args.mode == 'cremi-C':
        data_path = './data/cremi'
        f_raw = h5py.File(os.path.join(data_path, 'cremiC_inputs_interp.h5'), 'r')
        raw = f_raw['main'][:]
        f_raw.close()
        raw = raw[-50:]

        f_label = h5py.File(os.path.join(data_path, 'cremiC_labels.h5'), 'r')
        gt = f_label['main'][:]
        f_label.close()
        gt = gt[-50:]
    elif args.mode == 'cremi-B':
        data_path = './data/cremi'
        f_raw = h5py.File(os.path.join(data_path, 'cremiB_inputs_interp.h5'), 'r')
        raw = f_raw['main'][:]
        f_raw.close()
        raw = raw[-50:]

        f_label = h5py.File(os.path.join(data_path, 'cremiB_labels.h5'), 'r')
        gt = f_label['main'][:]
        f_label.close()
        gt = gt[-50:]
    elif args.mode == 'cremi-A':
        data_path = './data/cremi'
        f_raw = h5py.File(os.path.join(data_path, 'cremiA_inputs_interp.h5'), 'r')
        raw = f_raw['main'][:]
        f_raw.close()
        raw = raw[-50:]

        f_label = h5py.File(os.path.join(data_path, 'cremiA_labels.h5'), 'r')
        gt = f_label['main'][:]
        f_label.close()
        gt = gt[-50:]
    else:
        raise AttributeError('No this data mode!')
    gt = gt.astype(np.uint32)

    thresholds = np.arange(args.start_th, args.end_th+args.stride, args.stride)
    thresholds = list(thresholds)
    print('thresholds:', thresholds)

    # decide the index of th=0.5
    idx_th05 = -1
    for idx, th in enumerate(thresholds):
        if abs(th-0.5) < 0.000001:
            idx_th05 = idx
            break
    print('idx_th05: ', idx_th05)

    fragments = watershed(affs, 'maxima_distance')
    ### mask
    if args.mask_fragment is not None:
        tt = args.mask_fragment
        print('add mask and threshold=' + str(tt))
        affs_xy = 0.5 * (affs[1] + affs[2])
        fragments[affs_xy<tt] = 0

    # save fragments
    f_frag = h5py.File(os.path.join(out_affs, 'fragments.hdf'), 'w')
    f_frag.create_dataset('main', data=fragments, dtype=fragments.dtype, compression='gzip')
    f_frag.close()

    sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'

    seg = waterz.agglomerate(affs,
                        thresholds,
                        gt=gt,
                        fragments=fragments,
                        scoring_function=sf,
                        discretize_queue=256)

    best_arand = 1000
    best_idx = 0
    f_txt = open(os.path.join(out_affs, 'seg_waterz.txt'), 'w')
    seg_results = []
    for idx, seg_metric in enumerate(seg):
        segmentation = seg_metric[0].astype(np.int32)
        # metrics = seg_metric[1]
        # print('threshold=%.2f, voi_split=%.6f, voi_merge=%.6f, rand_split=%.6f, rand_merge=%.6f' % \
        #     (thresholds[idx], metrics['V_Info_split'], metrics['V_Info_merge'], metrics['V_Rand_split'], metrics['V_Rand_merge']))

        # segmentation = (segmentation * affs_xy).astype(np.int32)
        seg_results.append(segmentation)
        segmentation, _, _ = ev.relabel_from_one(segmentation)
        voi_merge, voi_split = ev.split_vi(segmentation, gt)
        voi_sum = voi_split + voi_merge
        arand = ev.adapted_rand_error(segmentation, gt)
        print('threshold=%.2f, voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
            (thresholds[idx], voi_split, voi_merge, voi_sum, arand))
        f_txt.write('threshold=%.2f, voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
            (thresholds[idx], voi_split, voi_merge, voi_sum, arand))
        f_txt.write('\n')
        if voi_sum < best_arand:
            best_arand = voi_sum
            best_idx = idx
    f_txt.close()
    print('Best threshold=%.2f, Best voi-sum=%.6f' % (thresholds[best_idx], best_arand))
    if idx_th05 != -1:
        best_idx = idx_th05
        print('save: ', best_idx)
    best_seg = randomlabel(seg_results[best_idx]).astype(np.uint16)
    f = h5py.File(os.path.join(out_affs, 'seg_waterz.hdf'), 'w')
    f.create_dataset('main', data=best_seg, dtype=np.uint16, compression='gzip')
    f.close()
    # show 
    if args.show:
        from utils.seeds_func import draw_fragments_3d
        print('show...')
        best_seg[gt == 0] = 0
        draw_fragments_3d(seg_img_path, best_seg, gt, raw)
    print('Done')
