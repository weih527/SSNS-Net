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
import multiprocessing
from joblib import Parallel
from joblib import delayed
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils.augmentation import SimpleAugment, RandomRotationAugment
from utils.augmentation import IntensityAugment, ElasticAugment
from utils.consistency_aug import resize_3d, gen_mask, add_gauss_noise, add_gauss_blur
from utils.seg_util import mknhood3d, genSegMalis
from utils.aff_util import seg_to_affgraph

class Train(Dataset):
	def __init__(self, cfg):
		super(Train, self).__init__()
		# multiprocess settings
		num_cores = multiprocessing.cpu_count()
		self.parallel = Parallel(n_jobs=num_cores, backend='threading')
		self.cfg = cfg
		self.model_type = cfg.MODEL.model_type

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
		self.if_scale_aug = cfg.DATA.if_scale_aug_labeled
		self.scale_factor = cfg.DATA.scale_factor
		self.if_filp_aug = cfg.DATA.if_filp_aug_labeled
		self.if_rotation_aug = cfg.DATA.if_rotation_aug_labeled
		self.if_intensity_aug = cfg.DATA.if_intensity_aug_labeled
		self.if_elastic_aug = cfg.DATA.if_elastic_aug_labeled
		self.if_noise_aug = cfg.DATA.if_noise_aug_labeled
		self.min_noise_std = cfg.DATA.min_noise_std
		self.max_noise_std = cfg.DATA.max_noise_std
		self.if_mask_aug = cfg.DATA.if_mask_aug_labeled
		self.if_blur_aug = cfg.DATA.if_blur_aug_labeled
		self.min_kernel_size = cfg.DATA.min_kernel_size
		self.max_kernel_size = cfg.DATA.max_kernel_size
		self.min_sigma = cfg.DATA.min_sigma
		self.max_sigma = cfg.DATA.max_sigma

		self.simple_aug = SimpleAugment()
		self.rotation_aug = RandomRotationAugment()
		self.intensity_aug = IntensityAugment()
		self.elastic_aug = ElasticAugment(control_point_spacing=[4, 40, 40],
										jitter_sigma=[0, 2, 2],
										padding=20)

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

			# assert self.dataset[0].shape[0] % 2 == 0, "the shape of raw data must be even"
			# padding_size_z = (self.crop_size[0] - self.dataset[0].shape[0]) // 2
			# for k in range(len(self.dataset)):
			# 	self.dataset[k] = np.pad(self.dataset[k], ((padding_size_z, padding_size_z), \
			# 											   (0, 0), \
			# 											   (0, 0)), mode='reflect')
			# 	self.labels[k] = np.pad(self.labels[k], ((padding_size_z, padding_size_z), \
			# 											   (0, 0), \
			# 											   (0, 0)), mode='reflect')

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

		# padding for random rotation
		self.crop_from_origin = [0, 0, 0]
		self.padding = 60
		if self.if_rotation_aug:
			self.crop_from_origin[0] = self.crop_size[0]
			self.crop_from_origin[1] = self.crop_size[1] + 2 * self.padding
			self.crop_from_origin[2] = self.crop_size[2] + 2 * self.padding
		else:
			self.crop_from_origin = self.crop_size

		# mask size
		if cfg.MODEL.model_type == 'mala':
			self.min_mask_size = [5, 5, 5]
			self.max_mask_size = [8, 12, 12]
			self.min_mask_counts = 40
			self.max_mask_counts = 60
		else:
			self.min_mask_size = [5, 10, 10]
			self.max_mask_size = [10, 20, 20]
			self.min_mask_counts = 60
			self.max_mask_counts = 100

	def __getitem__(self, index):
		# random select one dataset if contain many datasets
		k = random.randint(0, len(self.train_datasets)-1)
		used_data = self.dataset[k]
		used_label = self.labels[k]

		# random select one sub-volume
		if self.if_scale_aug:
			if random.random() > 0.5:
				min_size = self.crop_from_origin[-1] // self.scale_factor
				max_size = self.crop_from_origin[-1] * self.scale_factor
				det_size = random.randint(min_size // 2, max_size // 2)
				det_size = det_size * 2
				det_crop_size = [self.crop_from_origin[0], det_size, det_size]
			else:
				det_crop_size = self.crop_from_origin
		else:
			det_crop_size = self.crop_from_origin
		random_z = random.randint(0, self.raw_data_shape[0]-det_crop_size[0])
		random_y = random.randint(0, self.raw_data_shape[1]-det_crop_size[1])
		random_x = random.randint(0, self.raw_data_shape[2]-det_crop_size[2])
		imgs = used_data[random_z:random_z+det_crop_size[0], \
						random_y:random_y+det_crop_size[1], \
						random_x:random_x+det_crop_size[2]].copy()
		lb = used_label[random_z:random_z+det_crop_size[0], \
						random_y:random_y+det_crop_size[1], \
						random_x:random_x+det_crop_size[2]].copy()

		if imgs.shape[-1] != self.crop_from_origin[-1]:
			imgs = resize_3d(imgs, self.crop_from_origin[-1], mode='linear').astype(np.uint8)
			lb = resize_3d(lb, self.crop_from_origin[-1], mode='nearest').astype(np.uint16)

		# do augmentation
		imgs = imgs.astype(np.float32) / 255.0
		[imgs, lb] = self.simple_aug([imgs, lb])
		if self.if_intensity_aug:
			imgs = self.intensity_aug(imgs)
		if self.if_rotation_aug:
			imgs, lb = self.rotation_aug(imgs, lb)
			imgs = imgs[:, self.padding:-self.padding, self.padding:-self.padding]
			lb = lb[:, self.padding:-self.padding, self.padding:-self.padding]
		if self.if_elastic_aug:
			imgs, lb = self.elastic_aug(imgs, lb)
		if self.if_noise_aug:
			if random.random() > 0.5:
				imgs = add_gauss_noise(imgs, min_std=self.min_noise_std, max_std=self.max_noise_std, norm_mode='trunc')
		if self.if_blur_aug:
			if random.random() > 0.5:
				kernel_size = random.randint(self.min_kernel_size // 2, self.max_kernel_size // 2)
				kernel_size = kernel_size * 2 + 1
				sigma = random.uniform(self.min_sigma, self.max_sigma)
				imgs = add_gauss_blur(imgs, kernel_size=kernel_size, sigma=sigma)
		if self.if_mask_aug:
			if random.random() > 0.5:
				mask = gen_mask(imgs, model_type=self.model_type, \
								min_mask_counts=self.min_mask_counts, \
								max_mask_counts=self.max_mask_counts, \
								min_mask_size=self.min_mask_size, \
								max_mask_size=self.max_mask_size)
				imgs = imgs * mask

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

		# Norm images
		if self.if_norm_images:
			imgs = (imgs - 0.5) / 0.5
		# extend dimension
		imgs = imgs[np.newaxis, ...]
		imgs = np.ascontiguousarray(imgs, dtype=np.float32)
		lb_affs = np.ascontiguousarray(lb_affs, dtype=np.float32)
		weightmap = np.ascontiguousarray(weightmap, dtype=np.float32)
		return imgs, lb_affs, weightmap

	def __len__(self):
		return int(sys.maxsize)

def simple_augment(data, rule):
	assert np.size(rule) == 4
	assert data.ndim == 3
	# z reflection
	if rule[0]:
		data = data[::-1, :, :]
	# x reflection
	if rule[1]:
		data = data[:, :, ::-1]
	# y reflection
	if rule[2]:
		data = data[:, ::-1, :]
	# transpose in xy
	if rule[3]:
		data = np.transpose(data, (0, 2, 1))
	return data

def simple_augment_reverse(data, rule):
	assert np.size(rule) == 4
	assert len(data.shape) == 5
	# transpose in xy
	if rule[3]:
		# data = np.transpose(data, (0, 1, 2, 4, 3))
		data = data.permute(0, 1, 2, 4, 3)
	# y reflection
	if rule[2]:
		# data = data[:, :, :, ::-1, :]
		data = torch.flip(data, [3])
	# x reflection
	if rule[1]:
		# data = data[:, :, :, :, ::-1]
		data = torch.flip(data, [4])
	# z reflection
	if rule[0]:
		# data = data[:, :, ::-1, :, :]
		data = torch.flip(data, [2])
	return data

def simple_augment_debug(data, rule):
	assert np.size(rule) == 4
	assert data.ndim == 5
	# z reflection
	if rule[0]:
		data = data[:, :, ::-1, :, :]
	# x reflection
	if rule[1]:
		data = data[:, :, :, :, ::-1]
	# y reflection
	if rule[2]:
		data = data[:, :, :, ::-1, :]
	# transpose in xy
	if rule[3]:
		data = np.transpose(data, (0, 1, 2, 4, 3))
	return data

def collate_fn(batchs):
	out_input = []
	for batch in batchs:
		out_input.append(torch.from_numpy(batch['image']))
	
	out_input = torch.stack(out_input, 0)
	return {'image':out_input}

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

def show(img3d):
	# only used for image with shape [18, 160, 160]
	num = img3d.shape[0]
	column = 5
	row = math.ceil(num / float(column))
	size = img3d.shape[1]
	img_all = np.zeros((size*row, size*column), dtype=np.uint8)
	for i in range(row):
		for j in range(column):
			index = i*column + j
			if index >= num:
				img = np.zeros_like(img3d[0], dtype=np.uint8)
			else:
				img = (img3d[index] * 255).astype(np.uint8)
			img_all[i*size:(i+1)*size, j*size:(j+1)*size] = img
	return img_all


if __name__ == '__main__':
	import yaml
	from attrdict import AttrDict
	from utils.show import show_one
	""""""
	seed = 555
	np.random.seed(seed)
	random.seed(seed)
	cfg_file = 'seg_onlylb_suhu_snemi3d_data15.yaml'
	with open('./config/' + cfg_file, 'r') as f:
		cfg = AttrDict( yaml.load(f) )
	
	out_path = os.path.join('./', 'data_temp')
	if not os.path.exists(out_path):
		os.mkdir(out_path)
	data = Train(cfg)
	t = time.time()
	for i in range(0, 20):
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