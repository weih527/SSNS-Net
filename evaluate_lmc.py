import os 
import h5py
import argparse
import numpy as np

import elf.segmentation.multicut as mc
import elf.segmentation.features as feats
import elf.segmentation.watershed as ws
import evaluate as ev

def randomlabel(segmentation):
    segmentation = segmentation.astype(np.uint32)
    uid = np.unique(segmentation)
    mid = int(uid.max()) + 1
    mapping = np.zeros(mid, dtype=segmentation.dtype)
    mapping[uid] = np.random.choice(len(uid), len(uid), replace=False).astype(segmentation.dtype)#(len(uid), dtype=segmentation.dtype)
    out = mapping[segmentation]
    out[segmentation==0] = 0
    return out

def mc_baseline(affs, fragments=None):
    affs = 1 - affs
    boundary_input = np.maximum(affs[1], affs[2])
    if fragments is None:
        fragments = np.zeros_like(boundary_input, dtype='uint64')
        offset = 0
        for z in range(fragments.shape[0]):
            wsz, max_id = ws.distance_transform_watershed(boundary_input[z], threshold=.25, sigma_seeds=2.)
            wsz += offset
            offset += max_id
            fragments[z] = wsz
    rag = feats.compute_rag(fragments)
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    costs = feats.compute_affinity_features(rag, affs, offsets)[:, 0]
    edge_sizes = feats.compute_boundary_mean_and_length(rag, boundary_input)[:, 1]
    costs = mc.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes)
    node_labels = mc.multicut_kernighan_lin(rag, costs)
    segmentation = feats.project_node_labels_to_pixels(rag, node_labels)
    return segmentation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mn', '--model_name', type=str, default='None')
    parser.add_argument('-id', '--model_id', type=str, default=None)
    parser.add_argument('-m', '--mode', type=str, default='snemi3d-ac3')
    parser.add_argument('-ts', '--test_split', type=int, default=50)
    parser.add_argument('-nz', '--num_z', type=int, default=18)
    parser.add_argument('-mk', '--mask_fragment', type=float, default=None)
    parser.add_argument('-sw', '--show', action='store_true', default=False)
    args = parser.parse_args()
    
    trained_model = args.model_name
    out_path = os.path.join('./inference', trained_model, args.mode)
    img_folder = 'affs_'+args.model_id
    out_affs = os.path.join(out_path, img_folder)
    print('out_path: ' + out_affs)
    seg_img_path = os.path.join(out_affs, 'seg_lmc')
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

    # load fragments
    # f = h5py.File(os.path.join(out_affs, 'fragments.hdf'), 'r')
    # fragments = f['main'][:]
    # f.close()
    # seg = mc_baseline(affs, fragments=fragments)

    seg = mc_baseline(affs)

    f_txt = open(os.path.join(out_affs, 'seg_lmc.txt'), 'w')
    seg = seg.astype(np.int32)
    segmentation, _, _ = ev.relabel_from_one(seg)
    voi_merge, voi_split = ev.split_vi(segmentation, gt)
    voi_sum = voi_split + voi_merge
    arand = ev.adapted_rand_error(segmentation, gt)
    print('th=%.6f, voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
            (0.0, voi_split, voi_merge, voi_sum, arand))
    f_txt.write('th=%.6f, voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
            (0.0, voi_split, voi_merge, voi_sum, arand))
    f_txt.write('\n')
    f_txt.close()

    seg = randomlabel(seg).astype(np.uint16)
    f = h5py.File(os.path.join(out_affs, 'seg_lmc.hdf'), 'w')
    f.create_dataset('main', data=seg, dtype=np.uint16, compression='gzip')
    f.close()
    # show 
    if args.show:
        from utils.seeds_func import draw_fragments_3d
        print('show...')
        seg[gt == 0] = 0
        draw_fragments_3d(seg_img_path, seg, gt, raw)
    print('Done')