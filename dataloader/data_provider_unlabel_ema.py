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
		if cfg.DATA.unlabel_dataset == 'cremi-A-100':
			self.sub_path = 'cremi'
			self.train_datasets = ['cremiA_inputs_interp.h5']
		elif cfg.DATA.unlabel_dataset == 'cremi-A-200':
			self.sub_path = 'cremi'
			self.train_datasets = ['cremiA_inputs_interp.h5', 'cremiA+_inputs_interp.h5']
		elif cfg.DATA.unlabel_dataset == 'cremi-B-100':
			self.sub_path = 'cremi'
			self.train_datasets = ['cremiB_inputs_interp.h5']
		elif cfg.DATA.unlabel_dataset == 'cremi-B-200':
			self.sub_path = 'cremi'
			self.train_datasets = ['cremiB_inputs_interp.h5', 'cremiB+_inputs_interp.h5']
		elif cfg.DATA.unlabel_dataset == 'cremi-C-100':
			self.sub_path = 'cremi'
			self.train_datasets = ['cremiC_inputs_interp.h5']
		elif cfg.DATA.unlabel_dataset == 'cremi-C-200':
			self.sub_path = 'cremi'
			self.train_datasets = ['cremiC_inputs_interp.h5', 'cremiC+_inputs_interp.h5']
		elif cfg.DATA.unlabel_dataset == 'cremi-all':
			self.sub_path = 'cremi'
			self.train_datasets = ['cremiA_inputs_interp.h5', 'cremiB_inputs_interp.h5', 'cremiC_inputs_interp.h5']
		elif cfg.DATA.unlabel_dataset == 'snemi3d-ac3' or cfg.DATA.unlabel_dataset == 'snemi3d':
			self.sub_path = 'snemi3d'
			self.train_datasets = ['AC3_inputs.h5']
		elif cfg.DATA.unlabel_dataset == 'ac3_ac4':
			self.sub_path = 'snemi3d'
			self.train_datasets = ['AC3_inputs.h5', 'AC4_inputs.h5']
		elif cfg.DATA.unlabel_dataset == 'ac4_around':
			self.sub_path = 'snemi3d'
			self.train_datasets = ['AC3_inputs.h5', 'AC4_inputs.h5']
			self.train_datasets += list(cfg.DATA.unlabel_datalist)
		elif cfg.DATA.unlabel_dataset == 'fib-25' or cfg.DATA.unlabel_dataset == 'fib':
			self.sub_path = 'fib'
			self.train_datasets = ['fib_inputs.h5']
		else:
			raise AttributeError('No this dataset type!')

		# the path of datasets, need first-level and second-level directory, such as: os.path.join('../data', 'cremi')
		self.folder_name = os.path.join(cfg.DATA.data_folder, self.sub_path)

		# split unlabeled data
		# self.train_split = cfg.DATA.train_split
		self.unlabel_split = cfg.DATA.unlabel_split

		# augmentation
		self.if_norm_images = cfg.DATA.if_norm_images
		self.if_scale_aug = cfg.DATA.if_scale_aug_unlabel
		self.scale_factor = cfg.DATA.scale_factor
		self.if_filp_aug = cfg.DATA.if_filp_aug_unlabel
		self.if_rotation_aug = cfg.DATA.if_rotation_aug_unlabel
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

		self.simple_aug = SimpleAugment()
		self.rotation_aug = RandomRotationAugment()
		self.intensity_aug = IntensityAugment()
		self.elastic_aug = ElasticAugment(control_point_spacing=[4, 40, 40],
										jitter_sigma=[0, 2, 2],
										padding=20)

		# load dataset
		self.dataset = []
		for k in range(len(self.train_datasets)):
			print('load ' + self.train_datasets[k] + ' ...')
			# load raw data
			f_raw = h5py.File(os.path.join(self.folder_name, self.train_datasets[k]), 'r')
			data = f_raw['main'][:]
			f_raw.close()
			if data.shape[0] > self.unlabel_split:
				data = data[-self.unlabel_split:]
			self.dataset.append(data)

		# padding by 'reflect' mode for mala network
		if cfg.MODEL.model_type == 'mala':
			for k in range(len(self.dataset)):
				self.dataset[k] = np.pad(self.dataset[k], ((self.net_padding[0], self.net_padding[0]), \
														   (self.net_padding[1], self.net_padding[1]), \
														   (self.net_padding[2], self.net_padding[2])), mode='reflect')

		# the training dataset size
		self.raw_data_shape = list(self.dataset[0].shape)

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

		# random select one sub-volume
		if self.if_scale_aug:
			if random.random() > 0.5:
				min_size = self.crop_from_origin[-1] // self.scale_factor
				max_size = self.crop_from_origin[-1] * self.scale_factor
				det_size = random.randint(min_size // 2, max_size // 2)
				det_size = det_size * 2
				if det_size >= self.crop_from_origin[-1]:
					det_crop_size = [self.crop_from_origin[0], det_size, det_size]
				else:
					det_crop_size = self.crop_from_origin
			else:
				det_size = self.crop_from_origin[-1]
				det_crop_size = self.crop_from_origin
		else:
			det_size = self.crop_from_origin[-1]
			det_crop_size = self.crop_from_origin
		random_z = random.randint(0, self.raw_data_shape[0]-det_crop_size[0])
		random_y = random.randint(0, self.raw_data_shape[1]-det_crop_size[1])
		random_x = random.randint(0, self.raw_data_shape[2]-det_crop_size[2])
		imgs = used_data[random_z:random_z+det_crop_size[0], \
						random_y:random_y+det_crop_size[1], \
						random_x:random_x+det_crop_size[2]].copy()

		# simple aug
		[imgs] = self.simple_aug([imgs])

		if det_size == self.crop_from_origin[-1]:
			gt = imgs.copy()
		elif det_size > self.crop_from_origin[-1]:
			gt = imgs.copy()
			shift = (det_size - self.crop_from_origin[-1]) // 2
			gt = gt[:, shift:-shift, shift:-shift]
			imgs = resize_3d(imgs, self.crop_from_origin[-1], mode='linear').astype(np.uint8)
		else:
			gt = imgs.copy()
			shift = (self.crop_from_origin[-1] - det_size) // 2
			imgs = imgs[:, shift:-shift, shift:-shift]
			imgs = resize_3d(imgs, self.crop_from_origin[-1], mode='linear').astype(np.uint8)

		# if imgs.shape[-1] != self.crop_from_origin[-1]:
		# 	imgs = resize_3d(imgs, self.crop_from_origin[-1], mode='linear').astype(np.uint8)

		# do augmentation
		imgs = imgs.astype(np.float32) / 255.0
		gt = gt.astype(np.float32) / 255.0

		if self.if_intensity_aug:
			imgs = self.intensity_aug(imgs)
		if self.if_rotation_aug:
			imgs = self.rotation_aug(imgs)
			imgs = imgs[:, self.padding:-self.padding, self.padding:-self.padding]
		if self.if_elastic_aug:
			imgs = self.elastic_aug(imgs)
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

		# Norm images
		if self.if_norm_images:
			imgs = (imgs - 0.5) / 0.5
			gt = (gt - 0.5) / 0.5
		# extend dimension
		imgs = imgs[np.newaxis, ...]
		gt = gt[np.newaxis, ...]
		det_size = np.asarray([det_size], dtype=np.float32)
		det_size = det_size[np.newaxis, ...]
		imgs = np.ascontiguousarray(imgs, dtype=np.float32)
		gt = np.ascontiguousarray(gt, dtype=np.float32)
		rule = np.asarray([0,0,0,0], dtype=np.int32)
		rule = rule.astype(np.float32)
		return imgs, gt, det_size, rule

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
				batch[3] = batch[3].cuda()
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
				batch[3] = batch[3].cuda()
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
	cfg_file = 'seg_consist_suhu_snemi3d_d10_u200_dnoise.yaml'
	with open('./config/' + cfg_file, 'r') as f:
		cfg = AttrDict( yaml.load(f) )
	
	out_path = os.path.join('./', 'data_temp')
	if not os.path.exists(out_path):
		os.mkdir(out_path)
	data = Train(cfg)
	t = time.time()
	for i in range(0, 20):
		t1 = time.time()
		tmp_data, affs, det_size, _ = iter(data).__next__()
		# print('single cost time: ', time.time()-t1)
		print('det_size=%d' % det_size[0])
		tmp_data = np.squeeze(tmp_data)
		if cfg.MODEL.model_type == 'mala':
			tmp_data = tmp_data[14:-14,106:-106,106:-106]
		# affs_xy = affs[2]
		# weightmap_xy = weightmap[2]
		affs_xy = np.squeeze(affs)
		# affs_xy = (affs_xy-0.5) / 0.5

		img_data = show_one(tmp_data)
		img_affs = show_one(affs_xy)
		im_cat = np.concatenate([img_data, img_affs], axis=1)
		Image.fromarray(im_cat).save(os.path.join(out_path, str(i).zfill(4)+'.png'))
	print(time.time() - t)