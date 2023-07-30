# -*- coding: utf-8 -*-
import sys
import os

from PLNLDataSetClass import PLNLImageList

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import copy
import random
import numpy as np
import torch
from torchvision import transforms
from .datasets import register_dataset
import utils
from .prepocess import *

@register_dataset('Office31')
class Office31Dataset:
	"""
	OfficeHome Dataset class
	"""

	def __init__(self, name, img_dir, is_target):
		self.name = name
		self.img_dir = img_dir
		self.is_target = is_target

	def PLNLget_data(self):
		normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

		self.train_transforms = transforms.Compose([
			transforms.Resize(256),
			transforms.RandomCrop((224, 224)),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize_transform
		])
		self.test_transforms = transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			normalize_transform
		])

		# if self.LDS_type == 'natural':
		train_path = os.path.join('data/Office31/txt/', '{}.txt'.format(self.name))
		print(train_path)
		test_path = os.path.join('data/Office31/txt/', '{}.txt'.format(self.name))
		# elif self.LDS_type == 'RS_UT':
		# 	shift = 'UT' if self.is_target else 'RS'
		# 	train_path = os.path.join('data/OfficeHome/txt/', '{}_{}.txt'.format(self.name, shift))
		# 	test_path = os.path.join('data/OfficeHome/txt/', '{}_{}.txt'.format(self.name, shift))
		# else: raise NotImplementedError

		# return root/dog/xxx.png
		train_dataset = PLNLImageList(open(train_path).readlines(), os.path.join(self.img_dir, 'images'))
		# val_dataset = utils.ImageList(open(test_path).readlines(), os.path.join(self.img_dir, 'images'))
		# test_dataset = utils.ImageList(open(test_path).readlines(), os.path.join(self.img_dir, 'images'))


		train_dataset.targets = torch.from_numpy(train_dataset.labels)
		# train_dataset.targets, val_dataset.targets, test_dataset.targets = torch.from_numpy(train_dataset.labels), \
		# 																   torch.from_numpy(val_dataset.labels), \
		# 																   torch.from_numpy(test_dataset.labels)
		return train_dataset
