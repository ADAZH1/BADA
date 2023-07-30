import sys
import os
import pickle
import os
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from PLNLDataSetClass import PLNLImageList,PLNLDatasetWithIndicesWrapper
from PLNLdatasets import train_transform
from PLNLdatasets.datasets import get_PLNLdataset


def get_dataset(args, name, lbl_idx, unlbl_idx, pseudo_lbl_dict):

    train_lbl_idx = lbl_idx
    train_unlbl_idx = unlbl_idx

    lbl_idx = train_lbl_idx
    print(type(lbl_idx))
    if pseudo_lbl_dict is not None:
        print("pseudo_lbl_dict is Not None")
        pseudo_idx = pseudo_lbl_dict['pseudo_idx']
        pseudo_target = pseudo_lbl_dict['pseudo_target']
        nl_idx = pseudo_lbl_dict['nl_idx']
        nl_mask = pseudo_lbl_dict['nl_mask']
        # lbl_idx = lbl_idx.tolist()
        lbl_idx = np.array(lbl_idx + pseudo_idx)

        #if len(pseudo_idx) > 3000:
        #  print(a)

        #     balance the labeled and unlabeled data
        if len(nl_idx) > len(lbl_idx):
            exapand_labeled = len(nl_idx) // len(lbl_idx)
            lbl_idx = np.hstack([lbl_idx for _ in range(exapand_labeled)])

            if len(lbl_idx) < len(nl_idx):
                diff = len(nl_idx) - len(lbl_idx)
                lbl_idx = np.hstack((lbl_idx, np.random.choice(lbl_idx, diff)))
            else:
                assert len(lbl_idx) == len(nl_idx)

    else:
        print("pseudo label is None and nl_idx is None")
        pseudo_idx = None
        pseudo_target = None
        nl_idx = None
        nl_mask = None

    # print("train_lbl_dataset数据集初始化: ")
    # 融入现有的数据集加载方式
    is_target=True
    data_PLNLclass = get_PLNLdataset(name, name, args.img_dir, is_target)
    # print(lbl_idx)
    # train_lbl_dataset = data_PLNLclass.PLNLget_data(indexs = lbl_idx, pseudo_idx= pseudo_idx, pseudo_target= pseudo_target ,nl_idx= nl_idx, nl_mask= nl_mask)
    train_lbl_dataset = None
    train_unlbl_dataset = None
    train_nl_dataset = None
    train_lbl_dataset = data_PLNLclass.PLNLget_data()
    train_lbl_dataset = PLNLDatasetWithIndicesWrapper(name, train_lbl_dataset.data, train_lbl_dataset.targets, train_transform, pseudo_idx=pseudo_idx, pseudo_target=pseudo_target,
                                                    nl_idx=nl_idx, nl_mask=nl_mask, indexs=lbl_idx)

    if nl_idx is not None:
        print("nl is None")
        print("nl is None时的train_nl_dataset数据集初始化： ")
        train_nl_dataset = data_PLNLclass.PLNLget_data()

        # 10.6问题 唯一索引不太一样的是这边的 nl_idx?? 改动之前是转换成为numpy的
        train_nl_dataset = PLNLDatasetWithIndicesWrapper(name, train_nl_dataset.data, train_nl_dataset.targets, train_transform,
                                                         pseudo_idx = pseudo_idx, pseudo_target = pseudo_target,nl_idx = nl_idx, nl_mask = nl_mask, indexs = np.array(nl_idx))

  
    train_unlbl_dataset = data_PLNLclass.PLNLget_data()
    print("train_unlbl_dataset targets的长度："+str(len(train_unlbl_dataset.targets)))
    train_unlbl_dataset = PLNLDatasetWithIndicesWrapper(name, train_unlbl_dataset.data, train_unlbl_dataset.targets, train_transform,
                                                        pseudo_idx=pseudo_idx, pseudo_target=pseudo_target,nl_idx=nl_idx, nl_mask=nl_mask, indexs = train_unlbl_idx)
    print("train_unlbl_dataset的长度: " + str(len(train_unlbl_dataset)))
    # test_dataset = datasets.CIFAR10(root, train=False, transform=transform_val, download=True)

    if nl_idx is not None:
        print("nl_idx is Not None")
        # print("train_lbl_dataset样本长度: " + str(len(train_lbl_dataset)) + "train_unlbl_dataset样本长度: " + str(
        #     len(train_unlbl_dataset)))
        return train_lbl_dataset, train_nl_dataset, train_unlbl_dataset
    else:  # nl_idx is None    nl现在理解成negative label  使用负标签的数据集 一开始的话无法确认负标签 就直接返回Unlabel
        print("nl_idx is None")
        # print("train_lbl_dataset样本长度: " + str(len(train_lbl_dataset)) + "train_unlbl_dataset样本长度: " + str(
        #     len(train_unlbl_dataset)))
        return train_lbl_dataset, train_unlbl_dataset, train_unlbl_dataset
