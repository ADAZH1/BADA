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

    def get_PLNLdsets(self):
        """Generates and return train, val, and test datasets

        Returns:
            Train, val, and test datasets.
        """
        # 两个self.name figure out data_class's data type
        data_class = get_dataset(self.name, self.name, self.img_dir, self.is_target)

        self.num_classes, train_dataset, val_dataset, test_dataset, self.train_transforms, self.test_transforms = data_class.get_data()

        # train_dataset依然只是保存了数据集的路径和基本的transforms
        # def __init__(self, name, data, targets, transforms, base_transforms):

        return self.train_dataset, self.val_dataset, self.test_dataset



