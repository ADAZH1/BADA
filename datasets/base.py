# -*- coding: utf-8 -*-
import os
import sys
import copy
import random
import numpy as np
from PIL import Image
from collections import Counter
import pickle
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from .datasets import get_dataset, name2benchmark
from .prepocess import *
import utils
from RandAugment import RandAugment

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

sys.path.append('../')


class DatasetWithIndicesWrapper(torch.utils.data.Dataset):
    # def __init__(self, root, indexs, train=True,
    #              transform=None, target_transform=None,
    #              download=True, pseudo_idx=None, pseudo_target=None,
    #              nl_idx=None, nl_mask=None):
    #     super().__init__(root, train=train,
    #                      transform=transform,
    #                      target_transform=target_transform,
    #                      download=download)
    def __init__(self, name, data, targets, transforms, base_transforms):
        self.name = name
        self.data = data
        self.targets = targets
        self.transforms = transforms
        self.base_transforms = base_transforms
        self.rand_aug_transforms = copy.deepcopy(self.base_transforms)
        self.committee_size = 1
        self.ra_obj = RandAugment(1, 2.0)

        if self.name == 'mnist':  # See if this can be dropped entirely without losing performance
            self.transforms.transforms.insert(0, torchvision.transforms.Grayscale(num_output_channels=1))
            self.base_transforms.transforms.insert(0, torchvision.transforms.Grayscale(num_output_channels=1))
            self.rand_aug_transforms.transforms.insert(0, torchvision.transforms.Grayscale(num_output_channels=1))
        self.rand_aug_transforms.transforms.insert(0, self.ra_obj)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):

        data, target = self.data[index], self.targets[index]

        if self.name == 'mnist':
            data = Image.fromarray(data.numpy(), mode='L')
            data = data.convert('RGB')
        elif self.name == 'svhn':
            data = Image.fromarray(np.transpose(data, (1, 2, 0)))
        else:
            data = utils.default_loader(self.data[index])

        rand_aug_lst = [self.rand_aug_transforms(data) for _ in range(self.committee_size)]
        return (self.transforms(data), self.base_transforms(data), rand_aug_lst), int(target), int(index)


class ASDADataset:  # 将committe_size写入原来代码中
    def __init__(self, name, is_target=False, img_dir='data', valid_ratio=0.2, batch_size=128):
        self.name = name
        self.is_target = is_target
        self.img_dir = img_dir
        self.valid_ratio = valid_ratio
        self.batch_size = batch_size
        self.train_size = None

        self.train_dataset = None

        self.num_classes = None
        self.train_transforms = None
        self.test_transforms = None

    def get_num_classes(self):
        return self.num_classes

    def get_dsets(self):
        """Generates and return train, val, and test datasets

        Returns:
            Train, val, and test datasets.
        """
        # 两个self.name figure out data_class's data type
        data_class = get_dataset(self.name, self.name, self.img_dir, self.is_target)
        print("base中的get_data")
        self.num_classes, train_dataset, val_dataset, test_dataset, self.train_transforms, self.test_transforms = data_class.get_data()

        # train_dataset依然只是保存了数据集的路径和基本的transforms
        # def __init__(self, name, data, targets, transforms, base_transforms):
        self.train_dataset = DatasetWithIndicesWrapper(self.name, train_dataset.data, train_dataset.targets,
                                                       self.train_transforms, self.test_transforms)
        self.val_dataset = DatasetWithIndicesWrapper(self.name, val_dataset.data, val_dataset.targets,
                                                     self.test_transforms, self.test_transforms)
        self.test_dataset = DatasetWithIndicesWrapper(self.name, test_dataset.data, test_dataset.targets,
                                                      self.test_transforms, self.test_transforms)

        return self.train_dataset, self.val_dataset, self.test_dataset

    def get_loaders(self, shuffle=True, num_workers=4, class_balance_train=False):
        """Constructs and returns dataloaders

        Args:
            shuffle (bool, optional): Whether to shuffle dataset. Defaults to True.
            num_workers (int, optional): Number of threads. Defaults to 4.
            class_balance_train (bool, optional): Whether to class-balance train data loader. Defaults to False.

        Returns:
            Train, val, test dataloaders, as well as selected indices used for training
        """
        if not self.train_dataset: self.get_dsets()
        num_train = len(self.train_dataset)
        self.train_size = num_train

        benchmark = name2benchmark[self.name]

        if benchmark in ["MNIST", "SVHN"]:
            indices = list(range(num_train))
            split = int(np.floor(self.valid_ratio * num_train))
            if shuffle == True: np.random.shuffle(indices)
            train_idx, valid_idx = indices[split:], indices[:split]
            valid_sampler = SubsetRandomSampler(valid_idx)
        elif benchmark in ["DomainNet", "OfficeHome", "VisDA2017", "Office31"]:
            train_idx = np.arange(len(self.train_dataset))
            valid_sampler = SubsetRandomSampler(np.arange(len(self.val_dataset)))
        else:
            raise NotImplementedError

        train_sampler = SubsetRandomSampler(train_idx)

        # dataloader's function
        seq_loader = torch.utils.data.DataLoader(self.train_dataset, sampler=train_sampler, shuffle=False,
                                                 batch_size=self.batch_size, num_workers=num_workers)
        train_loader = torch.utils.data.DataLoader(self.train_dataset, sampler=train_sampler, \
                                                   batch_size=self.batch_size, num_workers=num_workers)

        val_loader = torch.utils.data.DataLoader(self.val_dataset, sampler=valid_sampler, \
                                                 batch_size=self.batch_size)
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)
        return train_loader, val_loader, test_loader, train_idx, seq_loader


