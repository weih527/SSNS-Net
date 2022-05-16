from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import cv2
import h5py
import math
import time
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils.augmentation import Rescale                        # with mask
from utils.augmentation import SimpleAugment as Filp          # with mask
from utils.consistency_aug_perturbations import Intensity
from utils.consistency_aug_perturbations import GaussBlur
from utils.consistency_aug_perturbations import GaussNoise
from utils.consistency_aug_perturbations import Cutout
from utils.consistency_aug_perturbations import SobelFilter
from utils.consistency_aug_perturbations import Mixup
from utils.augmentation import ElasticAugment as Elastic      # with mask
from utils.consistency_aug_perturbations import Artifact
from utils.consistency_aug_perturbations import Missing
from utils.consistency_aug_perturbations import BlurEnhanced

from utils.seg_util import mknhood3d, genSegMalis
from utils.aff_util import seg_to_affgraph

class Train(Dataset):
	def __init__(self, cfg):
		super(Train, self).__init__()
		self.cfg = cfg
		self.model_type = cfg.MODEL.model_type
		self.per_mode = cfg.DATA.per_mode

		# basic settings
		# the input size of network
		if cfg.MODEL.model_type == 'superhuman':
			self.crop_size = [18, 160, 160]
			self.net_padding = [0, 0, 0]
		elif cfg.MODEL.model_type == 'mala':
			self.crop_size = [53, 268, 268]
			self.net_padding = [14, 106, 106]  # the edge size of patch reduced by network
		else:
			raise AttributeError('No this model type!')

		# the output size of network
		# for mala: [25, 56, 56]
		# for superhuman: [18, 160, 160]
		self.out_size = [self.crop_size[k] - 2 * self.net_padding[k] for k in range(len(self.crop_size))]

		# training dataset files (h5), may contain many datasets
		if cfg.DATA.dataset_name == 'cremi-A' or cfg.DATA.dataset_name == 'cremi':
			self.sub_path = 'cremi'
			self.train_datasets = ['cremiA_inputs_interp.h5']
			self.train_labels = ['cremiA_labels.h5']
		elif cfg.DATA.dataset_name == 'cremi-B':
			self.sub_path = 'cremi'
			self.train_datasets = ['cremiB_inputs_interp.h5']
			self.train_labels = ['cremiB_labels.h5']
		elif cfg.DATA.dataset_name == 'cremi-C':
			self.sub_path = 'cremi'
			self.train_datasets = ['cremiC_inputs_interp.h5']
			self.train_labels = ['cremiC_labels.h5']
		elif cfg.DATA.dataset_name == 'cremi-all':
			self.sub_path = 'cremi'
			self.train_datasets = ['cremiA_inputs_interp.h5', 'cremiB_inputs_interp.h5', 'cremiC_inputs_interp.h5']
			self.train_labels = ['cremiA_labels.h5', 'cremiB_labels.h5', 'cremiC_labels.h5']
		elif cfg.DATA.dataset_name == 'snemi3d-ac3' or cfg.DATA.dataset_name == 'snemi3d':
			self.sub_path = 'snemi3d'
			self.train_datasets = ['AC3_inputs.h5']
			self.train_labels = ['AC3_labels.h5']
		elif cfg.DATA.dataset_name == 'snemi3d-ac4':
			self.sub_path = 'snemi3d'
			self.train_datasets = ['AC4_inputs.h5']
			self.train_labels = ['AC4_labels.h5']
		elif cfg.DATA.dataset_name == 'fib-25' or cfg.DATA.dataset_name == 'fib':
			self.sub_path = 'fib'
			self.train_datasets = ['fib_inputs.h5']
			self.train_labels = ['fib_labels.h5']
		else:
			raise AttributeError('No this dataset type!')

		# the path of datasets, need first-level and second-level directory, such as: os.path.join('../data', 'cremi')
		self.folder_name = os.path.join(cfg.DATA.data_folder, self.sub_path)
		assert len(self.train_datasets) == len(self.train_labels)

		# split training data
		self.train_split = cfg.DATA.train_split

		# augmentation
		self.if_norm_images = cfg.DATA.if_norm_images
		self.if_scale_aug = cfg.DATA.if_scale_aug_unlabel
		self.scale_factor = cfg.DATA.scale_factor
		self.if_filp_aug = False         # replace it with simple_aug
		self.if_rotation_aug = False     # replace it with simple_aug
		self.if_intensity_aug = cfg.DATA.if_intensity_aug_unlabel
		self.if_elastic_aug = cfg.DATA.if_elastic_aug_unlabel
		self.if_noise_aug = cfg.DATA.if_noise_aug_unlabel
		self.min_noise_std = cfg.DATA.min_noise_std
		self.max_noise_std = cfg.DATA.max_noise_std
		self.if_mask_aug = cfg.DATA.if_mask_aug_unlabel
		self.if_blur_aug = cfg.DATA.if_blur_aug_unlabel
		self.min_kernel_size = cfg.DATA.min_kernel_size
		self.max_kernel_size = cfg.DATA.max_kernel_size
		self.min_sigma = cfg.DATA.min_sigma
		self.max_sigma = cfg.DATA.max_sigma
		self.if_sobel_aug = cfg.DATA.if_sobel_aug_unlabel
		self.if_mixup_aug = cfg.DATA.if_mixup_aug_unlabel
		self.if_misalign_aug = cfg.DATA.if_misalign_aug_unlabel
		self.if_artifact_aug = cfg.DATA.if_artifact_aug_unlabel
		self.if_missing_aug = cfg.DATA.if_missing_aug_unlabel
		self.if_blurenhanced_aug = cfg.DATA.if_blurenhanced_aug_unlabel

		self.simple_aug = Filp()

		# load dataset
		self.dataset = []
		self.labels = []
		for k in range(len(self.train_datasets)):
			print('load ' + self.train_datasets[k] + ' ...')
			# load raw data
			f_raw = h5py.File(os.path.join(self.folder_name, self.train_datasets[k]), 'r')
			data = f_raw['main'][:]
			f_raw.close()
			data = data[:self.train_split]
			self.dataset.append(data)

			# load labels
			f_label = h5py.File(os.path.join(self.folder_name, self.train_labels[k]), 'r')
			label = f_label['main'][:]
			f_label.close()
			label = label[:self.train_split]
			self.labels.append(label)

		# padding when the shape(z) of raw data is smaller than the input of network
		numz_dataset = self.dataset[0].shape[0]
		if numz_dataset < self.crop_size[0]:
			padding_size_z_left = (self.crop_size[0] - numz_dataset) // 2
			if numz_dataset % 2 == 0:
				padding_size_z_right = padding_size_z_left
			else:
				padding_size_z_right = padding_size_z_left + 1
			for k in range(len(self.dataset)):
				self.dataset[k] = np.pad(self.dataset[k], ((padding_size_z_left, padding_size_z_right), \
															(0, 0), \
															(0, 0)), mode='reflect')
				self.labels[k] = np.pad(self.labels[k], ((padding_size_z_left, padding_size_z_right), \
															(0, 0), \
															(0, 0)), mode='reflect')

		# crop in xy plane for cremi dataset
		if cfg.DATA.label_crop_size is not None:
			label_crop_size = int(cfg.DATA.label_crop_size)
			for k in range(len(self.dataset)):
				depth, height, width = self.dataset[k].shape
				crop_shift = (height - label_crop_size) // 2
				self.dataset[k] = self.dataset[k][:, crop_shift:-crop_shift, crop_shift:-crop_shift]
				self.labels[k] = self.labels[k][:, crop_shift:-crop_shift, crop_shift:-crop_shift]

		# padding by 'reflect' mode for mala network
		if cfg.MODEL.model_type == 'mala':
			for k in range(len(self.dataset)):
				self.dataset[k] = np.pad(self.dataset[k], ((self.net_padding[0], self.net_padding[0]), \
														   (self.net_padding[1], self.net_padding[1]), \
														   (self.net_padding[2], self.net_padding[2])), mode='reflect')
				self.labels[k] = np.pad(self.labels[k], ((self.net_padding[0], self.net_padding[0]), \
														   (self.net_padding[1], self.net_padding[1]), \
														   (self.net_padding[2], self.net_padding[2])), mode='reflect')

		# the training dataset size
		self.raw_data_shape = list(self.dataset[0].shape)
		print('raw data shape: ', self.raw_data_shape)

		# padding for augmentation
		self.sub_padding = [0, 80, 80]  # for rescale
		self.crop_from_origin = [self.crop_size[i] + 2*self.sub_padding[i] for i in range(len(self.sub_padding))]

		# perturbations
		self.perturbations_init()

	def __getitem__(self, index):
		# random select one dataset if contain many datasets
		k = random.randint(0, len(self.train_datasets)-1)
		used_data = self.dataset[k]
		used_label = self.labels[k]

		# random select one sub-volume
		random_z = random.randint(0, self.raw_data_shape[0]-self.crop_from_origin[0])
		random_y = random.randint(0, self.raw_data_shape[1]-self.crop_from_origin[1])
		random_x = random.randint(0, self.raw_data_shape[2]-self.crop_from_origin[2])
		imgs = used_data[random_z:random_z+self.crop_from_origin[0], \
						random_y:random_y+self.crop_from_origin[1], \
						random_x:random_x+self.crop_from_origin[2]].copy()
		lb = used_label[random_z:random_z+self.crop_from_origin[0], \
						random_y:random_y+self.crop_from_origin[1], \
						random_x:random_x+self.crop_from_origin[2]].copy()

		# do augmentation
		imgs = imgs.astype(np.float32) / 255.0
		[imgs, lb] = self.simple_aug([imgs, lb])
		imgs, lb, _, _, _ = self.apply_perturbations(imgs, lb, None, mode=self.per_mode)

		# convert label to affinity
		if self.model_type == 'mala':
			lb = lb[self.net_padding[0]:-self.net_padding[0], \
					self.net_padding[1]:-self.net_padding[1], \
					self.net_padding[2]:-self.net_padding[2]]
		lb = genSegMalis(lb, 1)
		lb_affs = seg_to_affgraph(lb, mknhood3d(1), pad='replicate').astype(np.float32)

		# generate weights map for affinity
		weight_factor = np.sum(lb_affs) / np.size(lb_affs)
		weight_factor = np.clip(weight_factor, 1e-3, 1)
		weightmap = lb_affs * (1 - weight_factor) / weight_factor + (1 - lb_affs)

		# extend dimension
		imgs = imgs[np.newaxis, ...]
		imgs = np.ascontiguousarray(imgs, dtype=np.float32)
		lb_affs = np.ascontiguousarray(lb_affs, dtype=np.float32)
		weightmap = np.ascontiguousarray(weightmap, dtype=np.float32)
		return imgs, lb_affs, weightmap

	def perturbations_init(self):
		self.per_rescale = Rescale(scale_factor=self.scale_factor, det_shape=self.crop_size)
		self.per_flip = Filp()
		self.per_intensity = Intensity()
		self.per_gaussnoise = GaussNoise(min_std=self.min_noise_std, max_std=self.max_noise_std, norm_mode='trunc')
		self.per_gaussblur = GaussBlur(min_kernel=self.min_kernel_size, max_kernel=self.max_kernel_size, min_sigma=self.min_sigma, max_sigma=self.max_sigma)
		self.per_cutout = Cutout(model_type=self.model_type)
		self.per_sobel = SobelFilter(if_mean=True)
		self.per_mixup = Mixup(min_alpha=0.1, max_alpha=0.4)
		self.per_misalign = Elastic(control_point_spacing=[4, 40, 40], jitter_sigma=[0, 0, 0], prob_slip=0.2, prob_shift=0.2, max_misalign=17, padding=20)
		self.per_elastic = Elastic(control_point_spacing=[4, 40, 40], jitter_sigma=[0, 2, 2], padding=20)
		self.per_artifact = Artifact(min_sec=1, max_sec=5)
		self.per_missing = Missing(miss_fully_ratio=0.2, miss_part_ratio=0.5)
		self.per_blurenhanced = BlurEnhanced(blur_fully_ratio=0.5, blur_part_ratio=0.7)

	def apply_perturbations(self, data, mask, auxi=None, mode=1):
		all_pers = [self.if_scale_aug, self.if_filp_aug, self.if_rotation_aug, self.if_intensity_aug, \
					self.if_noise_aug, self.if_blur_aug, self.if_mask_aug, self.if_sobel_aug, \
					self.if_mixup_aug, self.if_misalign_aug, self.if_elastic_aug, self.if_artifact_aug, \
					self.if_missing_aug, self.if_blurenhanced_aug]
		if mode == 1:
			# select used perturbations
			used_pers = []
			for k, value in enumerate(all_pers):
				if value:
					used_pers.append(k)
			# select which one perturbation to use
			if len(used_pers) == 0:
				# do nothing
				# must crop
				data = data[:, self.sub_padding[-1]:-self.sub_padding[-1], self.sub_padding[-1]:-self.sub_padding[-1]]
				mask = mask[:, self.sub_padding[-1]:-self.sub_padding[-1], self.sub_padding[-1]:-self.sub_padding[-1]]
				scale_size = data.shape[-1]
				rule = np.asarray([0,0,0,0], dtype=np.int32)
				rotnum = 0
				return data, mask, scale_size, rule, rotnum
			elif len(used_pers) == 1:
				# No choise if only one perturbation can be used
				rand_per = used_pers[0]
			else:
				rand_per = random.choice(used_pers)
			# do augmentation
			# resize
			if rand_per == 0:
				data, mask, scale_size = self.per_rescale(data, mask)
			else:
				data = data[:, self.sub_padding[-1]:-self.sub_padding[-1], self.sub_padding[-1]:-self.sub_padding[-1]]
				mask = mask[:, self.sub_padding[-1]:-self.sub_padding[-1], self.sub_padding[-1]:-self.sub_padding[-1]]
				scale_size = data.shape[-1]
			# flip
			if rand_per == 1:
				# data, rule = self.per_flip(data)
				pass
			else:
				rule = np.asarray([0,0,0,0], dtype=np.int32)
			# rotation
			if rand_per == 2:
				# rotnum = random.randint(0, 3)
				# data = np.rot90(data, k=rotnum, axes=(1,2))
				pass
			else:
				rotnum = 0
			# intensity
			if rand_per == 3:
				data = self.per_intensity(data)
			# noise
			if rand_per == 4:
				data = self.per_gaussnoise(data)
			# blur
			if rand_per == 5:
				data = self.per_gaussblur(data)
			# mask or cutout
			if rand_per == 6:
				data = self.per_cutout(data)
			# sobel
			if rand_per == 7:
				data = self.per_sobel(data)
			# mixup
			if rand_per == 8 and auxi is not None:
				data = self.per_mixup(data, auxi)
			# misalign
			if rand_per == 9:
				# data = self.per_misalign(data)
				data, mask = self.per_misalign(data, mask)
			# elastic
			if rand_per == 10:
				# data = self.per_elastic(data)
				data, mask = self.per_elastic(data, mask)
			# artifact
			if rand_per == 11:
				data = self.per_artifact(data)
			# missing section
			if rand_per == 12:
				data = self.per_missing(data)
			# blur enhanced
			if rand_per == 13:
				data = self.per_blurenhanced(data)
		else:
			raise NotImplementedError
		return data, mask, scale_size, rule, rotnum

	def __len__(self):
		return int(sys.maxsize)


class Provider(object):
	def __init__(self, stage, cfg):
			#patch_size, batch_size, num_workers, is_cuda=True):
		self.stage = stage
		if self.stage == 'train':
			self.data = Train(cfg)
			self.batch_size = cfg.TRAIN.batch_size
			self.num_workers = cfg.TRAIN.num_workers
		elif self.stage == 'valid':
			# return valid(folder_name, kwargs['data_list'])
			pass
		else:
			raise AttributeError('Stage must be train/valid')
		self.is_cuda = cfg.TRAIN.if_cuda
		self.data_iter = None
		self.iteration = 0
		self.epoch = 1
	
	def __len__(self):
		return self.data.num_per_epoch
	
	def build(self):
		if self.stage == 'train':
			self.data_iter = iter(DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
                                             shuffle=False, drop_last=False, pin_memory=True))
		else:
			self.data_iter = iter(DataLoader(dataset=self.data, batch_size=1, num_workers=0,
                                             shuffle=False, drop_last=False, pin_memory=True))
	
	def next(self):
		if self.data_iter is None:
			self.build()
		try:
			batch = self.data_iter.next()
			self.iteration += 1
			if self.is_cuda:
				batch[0] = batch[0].cuda()
				batch[1] = batch[1].cuda()
				batch[2] = batch[2].cuda()
			return batch
		except StopIteration:
			self.epoch += 1
			self.build()
			self.iteration += 1
			batch = self.data_iter.next()
			if self.is_cuda:
				batch[0] = batch[0].cuda()
				batch[1] = batch[1].cuda()
				batch[2] = batch[2].cuda()
			return batch


if __name__ == '__main__':
	import yaml
	from attrdict import AttrDict
	from utils.show import show_one
	""""""
	seed = 555
	np.random.seed(seed)
	random.seed(seed)
	cfg_file = 'seg_snemi3d_d5_1024_u200.yaml'
	with open('./config/' + cfg_file, 'r') as f:
		cfg = AttrDict( yaml.load(f) )
	
	out_path = os.path.join('./', 'data_temp')
	if not os.path.exists(out_path):
		os.mkdir(out_path)
	data = Train(cfg)
	t = time.time()
	for i in range(0, 50):
		t1 = time.time()
		tmp_data, affs, weightmap = iter(data).__next__()
		print('single cost time: ', time.time()-t1)
		tmp_data = np.squeeze(tmp_data)
		if cfg.MODEL.model_type == 'mala':
			tmp_data = tmp_data[14:-14,106:-106,106:-106]
		affs_xy = affs[2]
		weightmap_xy = weightmap[2]

		img_data = show_one(tmp_data)
		img_affs = show_one(affs_xy)
		img_weight = show_one(weightmap_xy)
		im_cat = np.concatenate([img_data, img_affs, img_weight], axis=1)
		Image.fromarray(im_cat).save(os.path.join(out_path, str(i).zfill(4)+'.png'))
	print(time.time() - t)