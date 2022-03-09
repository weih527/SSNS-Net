import sys
import numpy as np
import h5py

from em_segLib.seg_malis import malis_init, malis_loss_weights_both
from em_segLib.seg_util import mknhood3d

# ---------------------
# 1. utility layers
class malisWeight():
    def __init__(self, conn_dims, opt_weight=0.5, opt_nb=1):
        # pre-compute 
        self.opt_weight=opt_weight
        if opt_nb==1:
            self.nhood_data = mknhood3d(1).astype(np.int32).flatten()
        else:
            self.nhood_data = mknhood3d(1,1.8).astype(np.uint64).flatten()
        self.nhood_dims = np.array((3,3),dtype=np.uint64)
        self.conn_dims = np.array(conn_dims[1:]).astype(np.uint64) # dim=4
        self.pre_ve, self.pre_prodDims, self.pre_nHood = malis_init(self.conn_dims, self.nhood_data, self.nhood_dims)
        self.weight = np.zeros(conn_dims,dtype=np.float32)#pre-allocate

    def getWeight(self, x_cpu, aff_cpu, seg_cpu):
        for i in range(x_cpu.shape[0]):
            self.weight[i] = malis_loss_weights_both(seg_cpu[i].flatten(), self.conn_dims, self.nhood_data, self.nhood_dims, self.pre_ve, self.pre_prodDims, self.pre_nHood, x_cpu[i].flatten(), aff_cpu[i].flatten(), self.opt_weight).reshape(self.conn_dims)
        return self.weight[:x_cpu.shape[0]]


# for L2 training: re-weight the error by label bias (far more 1 than 0)
class labelWeight():
    def __init__(self, conn_dims, opt_weight=2, clip_low=0.01, clip_high=0.99, thres=0.5):
        self.opt_weight=opt_weight
        self.clip_low=clip_low
        self.clip_high=clip_high
        self.thres = thres
        self.num_elem = np.prod(conn_dims[1:]).astype(float)
        self.weight = np.zeros(conn_dims,dtype=np.float32) #pre-allocate

    def getWeight(self, data):
        w_pos = self.opt_weight
        w_neg = 1.0-self.opt_weight
        for i in range(data.shape[0]):
            if self.opt_weight in [2,3]:
                frac_pos = np.clip(data[i].mean(), self.clip_low, self.clip_high) #for binary labels
                # can't be all zero
                w_pos = 1.0/(2.0*frac_pos)
                w_neg = 1.0/(2.0*(1.0-frac_pos))
                if self.opt_weight == 3: #match caffe param
                    w_pos = 0.5*w_pos**2
                    w_neg = 0.5*w_neg**2
            self.weight[i] = np.add((data[i] >= self.thres) * w_pos, (data[i] < self.thres) * w_neg)
        return self.weight/self.num_elem

def weightedMSE_np(input, target, weight=None, normalize_weight=False):
    # normalize by batchsize
    if weight is None:
        return np.sum((input - target) ** 2)/input.shape[0]
    else:
        if not normalize_weight: # malis loss: weight already normalized
            return np.sum(weight * (input - target) ** 2)/input.shape[0]
        else: # standard avg error
            return np.mean(weight * (input - target) ** 2)

def weightedMSE(input, target, weight=None, normalize_weight=False):
    import torch
    # normalize by batchsize
    if weight is None:
        return torch.sum((input - target) ** 2)/input.size(0)
    else:
        if not normalize_weight:
            return torch.sum(weight * (input - target) ** 2)/input.size(0)
        else:
            return torch.mean(weight * (input - target) ** 2)

