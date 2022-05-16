import os 
import h5py
import argparse
import numpy as np

import elf.segmentation.multicut as mc
import elf.segmentation.features as feats
import elf.segmentation.watershed as ws
import evaluate as ev

def mc_baseline(affs):
    affs = 1 - affs
    boundary_input = np.maximum(affs[1], affs[2])
    watershed = np.zeros_like(boundary_input, dtype='uint64')
    offset = 0
    for z in range(watershed.shape[0]):
        wsz, max_id = ws.distance_transform_watershed(boundary_input[z], threshold=.25, sigma_seeds=2.)
        wsz += offset
        offset += max_id
        watershed[z] = wsz
    rag = feats.compute_rag(watershed)
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    costs = feats.compute_affinity_features(rag, affs, offsets)[:, 0]
    edge_sizes = feats.compute_boundary_mean_and_length(rag, boundary_input)[:, 1]
    costs = mc.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes)
    node_labels = mc.multicut_kernighan_lin(rag, costs)
    segmentation = feats.project_node_labels_to_pixels(rag, node_labels)
    return segmentation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--in_path', type=str, default='', help='path to config file')
    parser.add_argument('-gt', '--gt_path', type=str, default='../data')
    parser.add_argument('-id', '--model_id', type=int, default=1000)
    parser.add_argument('-m', '--mode', type=str, default='ac3')
    parser.add_argument('-sn', '--seg_name', type=str, default='valid_seg')
    parser.add_argument('-ss', '--start_split', type=int, default=50)  # 25, 50, 75, 100
    parser.add_argument('-es', '--end_split', type=int, default=0)  # 25, 50, 75, 100
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

    seg = mc_baseline(affs)

    f_txt = open(os.path.join(affs_path, args.seg_name+'.txt'), 'a')
    seg = seg.astype(np.int32)
    segmentation, _, _ = ev.relabel_from_one(seg)
    voi_merge, voi_split = ev.split_vi(segmentation, test_label)
    voi_sum = voi_split + voi_merge
    arand = ev.adapted_rand_error(segmentation, test_label)
    print('model=%d, th=%.6f, voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
         (args.model_id, 0.0, voi_split, voi_merge, voi_sum, arand))
    f_txt.write('model=%d, th=%.6f, voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
                (args.model_id, 0.0, voi_split, voi_merge, voi_sum, arand))
    f_txt.write('\n')
    f_txt.close()