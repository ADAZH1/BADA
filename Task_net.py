import sys
import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import torch.nn.functional as F
from torchvision import models

import numpy as np

sys.path.append('../../')
import utils

np.random.seed(1234)
torch.manual_seed(1234)


class ResNet50Fc(nn.Module):

    def __init__(self, num_cls=1000, l2_normalize=False, temperature=1.0, bottleneck_dim=256):
        super(ResNet50Fc, self).__init__()
        self.model_resnet = models.resnet50(pretrained=True)

        model_resnet = self.model_resnet
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
        self.bn2 = nn.BatchNorm1d(bottleneck_dim)
        self.fc = nn.Linear(bottleneck_dim, num_cls)
        # self.fc = nn.Linear(model_resnet.fc.in_features, class_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.bottleneck(x)
        x = self.bn2(x)
        y = self.fc(x)
        return y

    def output_num(self):
        return self.__in_features

    def parameters_list(self, lr):
        parameter_list = [
            {'params': self.conv1.parameters(), 'lr': lr / 10},
            {'params': self.bn1.parameters(), 'lr': lr / 10},
            {'params': self.maxpool.parameters(), 'lr': lr / 10},
            {'params': self.layer1.parameters(), 'lr': lr / 10},
            {'params': self.layer2.parameters(), 'lr': lr / 10},
            {'params': self.layer3.parameters(), 'lr': lr / 10},
            {'params': self.layer4.parameters(), 'lr': lr / 10},
            {'params': self.avgpool.parameters(), 'lr': lr / 10},
            {'params': self.bottleneck.parameters()},
            # {'params': self.bn2.parameters()},
            {'params': self.fc.parameters()},
        ]

        return parameter_list