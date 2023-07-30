import copy
import pickle
import os
import numpy as np
import torch
import torchvision
from PIL import Image
from RandAugment import RandAugment
from torchvision import datasets
from torchvision import transforms

######################################################################
##### Data loading utilities
######################################################################
import utils


def PLNLdefault_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def PLNLmake_dataset(image_list, labels=None):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


class PLNLImageList(object):
    """A generic data loader where the images are arranged in this way: ::
		root/dog/xxx.png
		root/dog/xxy.png
		root/dog/xxz.png
		root/cat/123.png
		root/cat/nsdf3.png
		root/cat/asd932_.png
	Args:
		root (string): Root directory path.
		transform (callable, optional): A function/transform that  takes in an PIL image
			and returns a transformed version. E.g, ``transforms.RandomCrop``
		target_transform (callable, optional): A function/transform that takes in the
			target and transforms it.
		loader (callable, optional): A function to load an image given its path.
	 Attributes:
		classes (list): List of the class names.
		class_to_idx (dict): Dict with items (class_name, class_index).
		imgs (list): List of (image path, class_index) tuples
	"""

    def __init__(self, image_list, root, transform=None, target_transform=None,
                 loader=PLNLdefault_loader):
        imgs = PLNLmake_dataset(image_list)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"))

        self.root = root
        self.data = np.array([os.path.join(self.root, img[0]) for img in imgs])
        self.labels = np.array([img[1] for img in imgs])
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.data[index], self.labels[index]
        print(path)
        path = os.path.join(self.root, path)
        print("utils'path: " + str(path))
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,  # indexs是train_unlbl_idx
                 transform=None, target_transform=None,
                 download=True, pseudo_idx=None, pseudo_target=None,
                 nl_idx=None, nl_mask=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.targets = np.array(self.targets)
        self.nl_mask = np.ones((len(self.targets), len(np.unique(self.targets))))  # 样本数目*类别数？

        if nl_mask is not None:
            self.nl_mask[nl_idx] = nl_mask  # 可以参与 负学习过程的下标

        if pseudo_target is not None:
            self.targets[pseudo_idx] = pseudo_target  # pseudo_idx被选择成为伪标签的下标

        if indexs is not None:  # 这边代表的Index代表labeled
            indexs = np.array(indexs)
            print("indexs is not None")
            # print(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.nl_mask = np.array(self.nl_mask)[indexs]
            self.indexs = indexs
        else:
            print("index is None")
            self.indexs = np.arange(len(self.targets))
            # print(self.indexs)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.indexs[index], self.nl_mask[index]
        # return img, target, index, self.nl_mask[index]


class PLNLDatasetWithIndicesWrapper(torch.utils.data.Dataset):

    def __init__(self, name, data, targets, transforms, pseudo_idx=None,
                 pseudo_target=None,
                 nl_idx=None, nl_mask=None, indexs=None):
        self.name = name
        self.data = data
        self.targets = np.array(targets)
        self.transforms = transforms
        self.nl_mask = np.ones((len(self.targets), len(np.unique(self.targets))))  # 样本数目*类别数？

        # 这边应该是无所谓当时选择哪个的
        self.committee_size = 3
        self.rand_aug_transforms = copy.deepcopy(self.transforms)
        self.ra_obj = RandAugment(3, 2.0)
        self.rand_aug_transforms.transforms.insert(0, self.ra_obj)

        # print("self.nl_mask的形状： " + str(self.nl_mask.shape))
        if nl_mask is not None:
            # print("nl_idx的类型： " + str(type(nl_idx)))
            # print("nl_idx的形状： " + str(len(nl_idx)))
            # # print(nl_idx)
            #
            # print("nl_mask的类型： " + str(type(nl_mask)))
            # print("nl_mask的形状： " + str(len(nl_mask)))
            # # print(nl_mask)

            self.nl_mask[nl_idx] = (nl_mask)  # 可以参与 负学习过程的下标
            # print("self.nl_mask[nl_idx]的形状： " + str(self.nl_mask[nl_idx].shape))
        if pseudo_target is not None:
            self.targets[pseudo_idx] = pseudo_target  # pseudo_idx被选择成为伪标签的下标

        if indexs is not None:  # 这边代表的Index代表labeled
            indexs = np.array(indexs)
            # print("indexs is not None")
            # print(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.nl_mask = np.array(self.nl_mask)[indexs]
            self.indexs = indexs
        else:
            # print("index is None")
            self.indexs = np.arange(len(self.targets))
            # print(self.indexs)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = PLNLdefault_loader(self.data[index])
        # if self.transforms is not None:
        #     img = self.transforms(img)

        rand_aug_lst = [self.rand_aug_transforms(img) for _ in range(self.committee_size)]
        return (self.transforms(img), rand_aug_lst), target, self.indexs[index], self.nl_mask[index]
        # (self.transforms(data), self.base_transforms(data), rand_aug_lst), int(target), int(index)
        # return img, target, index, self.nl_mask[index]

    def __len__(self):
        return len((self.data))