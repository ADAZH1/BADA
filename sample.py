# -*- coding: utf-8 -*-
"""
Implements active learning sampling strategies
Adapted from https://github.com/ej0cl6/deep-active-learning
"""

import os
import copy
import random

import numpy
import numpy as np

import scipy
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import argsort
from torchvision import datasets
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from tqdm import tqdm
import datetime
import utils
from utils import ActualSequentialSampler
from utils import *
from adapt.solvers.solver import get_solver

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

al_dict = {}


def register_strategy(name):
    def decorator(cls):
        al_dict[name] = cls
        return cls

    return decorator


def get_strategy(sample, *args):
    if sample not in al_dict: raise NotImplementedError
    return al_dict[sample](*args)


class SamplingStrategy:
    """
    Sampling Strategy wrapper class
    """

    def __init__(self, src_train_loader, dset, tgt_loader, train_idx, model, discriminator,device, src_seq_loader,args):
        self.src_train_loader = src_train_loader
        self.dset = dset
        self.tgt_loader = tgt_loader
        if dset.name == 'DomainNet':
            self.num_classes = self.dset.get_num_classes()
        else:
            self.num_classes = len(set(dset.targets.numpy()))
        self.train_idx = np.array(train_idx)
        self.model = model
        self.device = device
        self.discriminator = discriminator
        self.args = args
        self.idxs_lb = np.zeros(len(self.train_idx), dtype=bool)
        self.src_seq_loader = src_seq_loader

    def query(self, n):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb
        # print("当前的self.idxs_lb" + str(self.idxs_lb))

    def train(self, target_train_dset, da_round=1, src_loader=None, src_model=None):
        """
        Driver train method
        """
        best_val_acc, best_model = 0.0, None
        print("da_round: " + str(da_round))
        train_sampler = SubsetRandomSampler(self.train_idx[self.idxs_lb])


        tgt_sup_loader = torch.utils.data.DataLoader(target_train_dset, sampler=train_sampler, num_workers=4, \
                                                     batch_size=self.args.batch_size, drop_last=False)  # 目标域有监督的样本
        print(len(tgt_sup_loader.sampler)) # 所以采样的其实只有十个sample
        print(tgt_sup_loader.dataset.committee_size)
        print(src_loader.dataset.committee_size)
        tgt_unsup_loader = torch.utils.data.DataLoader(target_train_dset, shuffle=True, num_workers=4, \
                                                       batch_size=self.args.batch_size, drop_last=False)
        # opt_net_tgt = optim.Adam(self.model.parameters(), lr=self.args.adapt_lr, weight_decay=self.args.wd, betas=(0.9, 0.999))

        # Update discriminator adversarially with classifier
        # lr_scheduler = optim.lr_scheduler.StepLR(opt_net_tgt, 20, 0.5)

        opt_net = utils.generate_optimizer(self.model, self.args, mode='da')
        # opt_net = optim.Adadelta(self.model.parameters_list(self.args.lr), lr=self.args.lr)  # resnet50的优化器
        solver = get_solver(self.args.da_strat, self.model, src_loader, tgt_sup_loader, tgt_unsup_loader, \
                            self.train_idx, opt_net, da_round, self.device, self.args)

        for epoch in range(self.args.adapt_num_epochs):
            if self.args.da_strat == 'dann':
                # opt_dis_adapt = optim.SGD(self.discriminator.parameters(), lr=(self.args.lr) / 10,
                #                           weight_decay=0.0005)
                opt_dis_adapt = optim.SGD(self.discriminator.parameters(), lr=(self.args.lr) / 10,
                                          weight_decay=0.0005)
                solver.solve(epoch, self.discriminator, opt_dis_adapt)
            elif self.args.da_strat in ['ft', 'mme','Tradmme']:
                solver.solve(epoch)
            else:
                raise NotImplementedError


        return self.model, opt_net

    def PseudoBasedRank(self, model, dataloader):
        model.eval()
        stat = list()

        # 获得形如64*10的数据（64代表batchsize 10则是类别 每个类别都是由标准差算出）
        # all_log_probs, all_scores = [], []
        # all_log_probs = torch.zeros([len(dataloader.sampler), self.num_classes])  # 4365*65

        log_prob = torch.zeros([len(dataloader.sampler), self.num_classes]).float().to(self.device)  # 4365 * 65
        unc = torch.zeros(len(dataloader.sampler)).float().to(self.device)

        with torch.no_grad():
            # for batch_idx, ((data, _), target, _) in enumerate(tqdm(train_loader)):
            for batch_idx, ((_, data_t_og, data_t_raug), label_t, index) in enumerate(tqdm(dataloader)):
                data = data_t_og.to(self.device)
                label_t = label_t.to(self.device)
                y = model(data)

                # Fist Criterion: Uncertainty based on data augmentation's consistency
                data_committee = np.zeros((self.committee_size, data.shape[0], self.num_classes)) # 3*64(batch_size)*65
                data_committee = torch.from_numpy(data_committee)
                data_committee_count = 0    # 计算
                for data_t_aug_curr in data_t_raug:  # 遍历每个数据的增广结果
                    score_t_aug_curr = model(data_t_aug_curr.to(self.device))
                    data_committee[data_committee_count] = score_t_aug_curr
                    data_committee_count = data_committee_count + 1


                uncertainty , d = get_Uncertainty(y, data_committee[0], data_committee[1], data_committee[2],
                                              self.device) # uncertainty: 32      d: 32 * 65
                # print("uncertainty: " + str(uncertainty))
                unc[index] = uncertainty
                log_prob[index, :] = d
                # pred = y.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                # correct = pred.eq(label_t.view_as(pred))  # correct's shape
                # for i in range(len(correct)):
                #     # print("uncertainty[i]: " + str(i) + str(uncertainty[i].item()))
                #     stat.append([indices_t[i].item(), uncertainty[i].item()])
                #     # print(stat[i])

        return unc, log_prob

@register_strategy('SENTRYTest')
class SENTRYTestSampling(SamplingStrategy):
    """
    Implements CLUE: CLustering via Uncertainty-weighted Embeddings
    """

    #  def __init__(self, dset, tgt_loader, train_idx, model, device, args, balanced=False):
    def __init__(self, src_train_loader, dset, tgt_loader, train_idx, model,  discriminator ,device, src_seq_loader,args):
        super(SENTRYTestSampling, self).__init__(src_train_loader, dset, tgt_loader, train_idx, model, discriminator ,device, src_seq_loader,args)
        # Committee consistency hyperparameters
        self.randaug_n = 3  # RandAugment number of consecutive transformations
        self.randaug_m = 2.0  # RandAugment severity
        self.committee_size = 3  # Committee size

        # Pass in hyperparams to dataset
        self.tgt_loader.dataset.committee_size = self.committee_size
        self.tgt_loader.dataset.ra_obj.n = self.randaug_n
        self.tgt_loader.dataset.ra_obj.m = self.randaug_m

        # self.T = self.args.clue_softmax_t

        # train_loader打乱transform之后 每次取一批次 是否还是同一幅照片的不同形式

    def query(self, n):  # n是选取的
        print("train_idx长度: " + str(len(self.train_idx)))
        # self.idxs_lb = np.zeros(len(self.train_idx), dtype=bool) idxs_lb index labeled 初始全部赋值为0 因为初始时候是没有样本会被标注的

        # return np.random.choice(np.where(self.idxs_lb == 0)[0], n, replace=False)
        idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]  #
        train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])

        print("self.train_idx[idxs_unlabled]的长度: " + str(
            len(self.train_idx[idxs_unlabeled])))  # 这边实现采样是每次都要重新K-MEANS 选取样本
        data_loader = torch.utils.data.DataLoader(self.dset, sampler=train_sampler, num_workers=4, \
                                                  batch_size=self.args.batch_size, drop_last=False)
        print("data_loader中train_sampler的长度： " + str(len(data_loader.sampler)))  # 根据idxs_unlabeled选取样本

        if self.args.cnn == 'LeNet':
            emb_dim = 500
        elif self.args.cnn == 'ResNet50':
            emb_dim = 32
        device = self.device
        model = self.model

        # Firstly, for each sample, compute its uncertainty
        # 按照TQS的思路，输入每个train_loader计算出 每个样本的一个得分 主要就是下标的记录
        model.eval()
        stat = list()

        all_log_probs, all_scores = [], []

        # log_prob = torch.zeros([len(data_loader.sampler), self.num_classes]).double().to(device)  # 4365 * 65
        # unc = torch.zeros(len(data_loader.sampler)).double().to(device)
        # 保存前两次的标准差看一下
        # remark0 = torch.zeros((self.num_classes)).double()
        # remark1 = torch.zeros((self.num_classes)).double()
        with torch.no_grad():
            for batch_idx, ((_, data_t_og, data_t_raug), label_t, index) in enumerate(tqdm(data_loader)):
                # print(index)
                data = data_t_og.to(device)
                label_t = label_t.to(device)
                y = model(data)
                # print((y.argmax(dim=1, keepdim=True)).shape)
                # preds[index] = y.argmax(dim=1, keepdim=True).squeeze()
                # print("preds: " + str(preds.shape))

                # Fist Criterion: Uncertainty based on data augmentation's consistency
                data_committee = np.zeros((self.committee_size, data.shape[0], self.num_classes))
                data_committee = torch.from_numpy(data_committee)
                data_committee_count = 0  # 计算
                for data_t_aug_curr in data_t_raug:  # 遍历每个数据的增广结果
                    score_t_aug_curr = model(data_t_aug_curr.to(self.device))
                    data_committee[data_committee_count] = score_t_aug_curr
                    data_committee_count = data_committee_count + 1
                # print(data_committee[0].shape) # 32 * 65
                uncertainty, d = get_Uncertainty(y, data_committee[0], data_committee[1], data_committee[2],
                                                 self.device)  # uncertainty: 32      d: 32 * 65

                # if index[0] == 0:
                #     # print("d[0]: " + str(d[0]))
                #     remark0 = d[0]
                # if index[1] == 1:
                #     # print(d[1])
                #     remark1 = d[1]

                all_scores.append(uncertainty)
                # pred = y.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                # correct = pred.eq(label_t.view_as(pred))  # correct's shape
                # for i in range(len(correct)):
                #     # print("uncertainty[i]: " + str(i) + str(uncertainty[i].item()))
                #     stat.append([indices_t[i].item(), uncertainty[i].item()])
                #     # print(stat[i])

        # print("log_prob的前两次标准差结果： ")
        # print(log_prob[0])
        # print("实验中 0 ")
        # print(remark0)
        # print(log_prob[1])
        # print("实验中 1")
        # print(remark1)
        # print("unc: " + str(unc.shape))
        # print("log_prob: " + str(log_prob.shape))


        all_scores = torch.cat(all_scores)
        scores = (all_scores).sort(descending=True)[1]
        top_N = int(len(scores) * 0.05) #
        #top_N = int(len(scores) * 1.0)
        q_idxs = np.random.choice(scores[:top_N].cpu().numpy(), n, replace=False)
        return idxs_unlabeled[q_idxs]
        
        
        # 
        # VisDA分界线
    
    
    
        # Neighborhood Uncertainty
        # tgt_emb, tgt_lab, tgt_preds, tgt_pen_emb
        # src_emb, src_labels, src_preds, src_pen_emb = utils.get_embedding(self.model, self.src_seq_loader,
        #                                                                   self.device, \
        #                                                                   self.num_classes, \
        #                                                                   self.args, with_emb=True, emb_dim=emb_dim)
        # print(src_labels)
        # matrix = torch.cdist(src_pen_emb, tgt_pen_emb)
        # matrix = matrix + 1.0
        # matrix = 1.0 / matrix
        #
        # ind = torch.sort(matrix, descending=True, dim=0).indices # 这边已经确定是找最近的了
        # ind_split = torch.split(ind, 1, dim=1)
        # print("ind_split的长度是； " + str(len(ind_split)))
        # print("target_dataloader的长度是: " + str(len(data_loader.sampler)))
        # ind_split = [id.squeeze() for id in ind_split]
        # # tgt_lab  直接就是伪标签
        #
        # confidence = []
        # sim_matrix_split = torch.split(matrix, 1, dim=1)
        # sim_matrix_split = [_id.squeeze() for _id in sim_matrix_split]
        # for i in range(0, len(data_loader.sampler)):
        #     _row = ind_split[i].long()  # 第i个源域样本在目标域最近的编号  不对
        #     #  上面dim=0排序之后  应该是第i个目标域样本在源域中距离最近的样本编号
        #     sim_scoreTemp = 0.0
        #     all_scoreTemp = 0.0
        #     for j in range(0, int(len(self.src_seq_loader.sampler)/65) + 5):
        #         if src_labels[_row[j].item()].item() != tgt_lab[i].item():
        #             sim_scoreTemp += sim_matrix_split[i][_row[j].item()].item()
        #
        #         all_scoreTemp += sim_matrix_split[i][_row[j].item()].item()
        #     ratio = sim_scoreTemp / all_scoreTemp
        #     ratio = torch.Tensor([ratio])
        #     confidence.append(ratio)
        # confidence = torch.cat(confidence)
        # confidence = confidence.to(device)
        #
        # #  Sample from top-2 % instances, as recommended by authors
        #
        # # Get diversity and entropy
        # all_log_probs, all_scores = [], []
        # with torch.no_grad():
        #     for batch_idx, ((data, _, _), target, indices) in enumerate(data_loader):  # 这边取出的data_loader的index肯定就是连续的
        #         data, target = data.to(self.device), target.to(self.device)
        #         scores = self.model(data)
        #         log_probs = nn.LogSoftmax(dim=1)(scores)
        #         all_scores.append(scores)
        #         all_log_probs.append(log_probs)
        #
        # all_scores = torch.cat(all_scores)
        # all_log_probs = torch.cat(all_log_probs)
        #
        # all_probs = torch.exp(all_log_probs)
        # disc_scores = nn.Softmax(dim=1)(self.discriminator(all_scores))
        # Entropy = -(all_probs * all_log_probs).sum(1)
        # Entropy = Entropy.to(device)
        # scores = (0.6 * DivScore + 0.4 * confidence).sort(descending=True)[1]
        # scores = (DivScore).sort(descending=True)[1]
        # top_N = int(len(scores) * 0.05)
        # q_idxs = np.random.choice(scores[:top_N].cpu().numpy(), n, replace=False)
        #
        # print("q_idxs的内容是: " + str(q_idxs))
        # return idxs_unlabeled[q_idxs]


@register_strategy('uniform')
class RandomSampling(SamplingStrategy):
    """
    Uniform sampling
    """

    # def __init__(self, dset, dset1, dset2, dset3, dset4, dset5, train_idx, model, device, args, balanced=False):
    #     super(SENTRYDerivedSampling, self).__init__(dset, dset1, dset2, dset3, dset4, dset5, train_idx, model, device,
    #                                                 args)
    #     self.random_state = np.random.RandomState(1234)
    
    def __init__(self, src_train_loader, dset, tgt_loader, train_idx, model, discriminator ,device, src_seq_loader, args):
        super(RandomSampling, self).__init__( src_train_loader, dset, tgt_loader, train_idx, model, discriminator ,device, src_seq_loader, args)
        self.labels = dset.labels if dset.name == 'DomainNet' else dset.targets
        self.classes = np.unique(self.labels)
        self.dset = dset


    def query(self, n):
        return np.random.choice(np.where(self.idxs_lb == 0)[0], n, replace=False)


@register_strategy('SENTRYDerived')
class SENTRYDerivedSampling(SamplingStrategy):
    """
    Implements CLUE: CLustering via Uncertainty-weighted Embeddings
    """

    # def __init__(self, src_train_loader, dset, tgt_loader, train_idx, model, discriminator, device, args):
    #  def __init__(self, dset, tgt_loader, train_idx, model, device, args, balanced=False):

    def __init__(self, src_train_loader, dset, tgt_loader, train_idx, model,  discriminator ,device, src_seq_loader,args):
        super(SENTRYDerivedSampling, self).__init__(src_train_loader, dset, tgt_loader, train_idx, model, discriminator ,device, src_seq_loader,args)
        # Committee consistency hyperparameters
        # self.randaug_n = 3  # RandAugment number of consecutive transformations
        # self.randaug_m = 2.0  # RandAugment severity
        # self.committee_size = 3  # Committee size
        #
        # # Pass in hyperparams to dataset
        # self.tgt_loader.dataset.committee_size = self.committee_size
        # self.tgt_loader.dataset.ra_obj.n = self.randaug_n
        # self.tgt_loader.dataset.ra_obj.m = self.randaug_m

        # self.T = self.args.clue_softmax_t

    # train_loader打乱transform之后 每次取一批次 是否还是同一幅照片的不同形式
    def query(self, n):  # n是选取的
        print("train_idx长度: " + str(len(self.train_idx)))
        # self.idxs_lb = np.zeros(len(self.train_idx), dtype=bool) idxs_lb index labeled 初始全部赋值为0 因为初始时候是没有样本会被标注的

        # return np.random.choice(np.where(self.idxs_lb == 0)[0], n, replace=False)

        idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb] #

        train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])

        print("self.train_idx[idxs_unlabled]的长度: " + str(
            len(self.train_idx[idxs_unlabeled])))  # 这边实现采样是每次都要重新K-MEANS 选取样本
        data_loader = torch.utils.data.DataLoader(self.dset, sampler=train_sampler, num_workers=4, \
                                                  batch_size=self.args.batch_size, drop_last=False)
        print("data_loader中train_sampler的长度： " + str(len(data_loader.sampler))) #根据idxs_unlabeled选取样本

        if self.args.cnn == 'LeNet':
            emb_dim = 500
        elif self.args.cnn == 'ResNet50':
            emb_dim = 32
        device = self.device
        model = self.model

        # Firstly, for each sample, compute its uncertainty
        # 按照TQS的思路，输入每个train_loader计算出 每个样本的一个得分 主要就是下标的记录
        model.eval()
        stat = list()

        tgt_emb, tgt_lab, tgt_preds, tgt_pen_emb = utils.get_embedding(self.model, data_loader, self.device, \
                                                                       self.num_classes, \
                                                                       self.args, with_emb=True, emb_dim=emb_dim)
        print("tgt_pen_emb的形状： " + str(tgt_pen_emb.shape))

        # tgt_emb: 分类概率  tgt_lab:target labels 标签？  其实主要用到的就是tgt_emb和tgt_pen_emb  tgt_pen_emb代表样本的特征空间 shape:[len(sampler)*emb_dim]
        start1 = datetime.datetime.now()
        dist = torch.zeros((tgt_pen_emb.shape[0], tgt_pen_emb.shape[0]))  # 所有样本之间的距离矩阵
        dist = dist.to(device)
        # for i in range(0, tgt_pen_emb.shape[0]): # 计算目标域样本内部相互之间的余弦相似度
        #     for j in range(0, tgt_pen_emb.shape[0]):
        #         dist[i][j] = F.cosine_similarity(tgt_pen_emb[i].unsqueeze(0), tgt_pen_emb[j].unsqueeze(0))
        #
        dist = torch.cdist(tgt_pen_emb, tgt_pen_emb)
        end1 = datetime.datetime.now()
        print("dist的形状："  + str(dist.shape))
        print(end1-start1)

        start2 = datetime.datetime.now()
        DivScore = torch.zeros(dist.shape[0])
        SortDist, indices = torch.sort(dist, descending=True)
        DivScore, SortDist = DivScore.to(device), SortDist.to(device)
        for i in range(0, SortDist.shape[0]):
            for j in range(0, 38):
                DivScore[i] += SortDist[i][j]
        end2 = datetime.datetime.now()
        print(end2 - start2)
        # 上面一步只是确实距离 后面需要乘熵 上面关键还在于i和sample_weights中的index

        # Neighborhood Uncertainty
        # tgt_emb, tgt_lab, tgt_preds, tgt_pen_emb
        src_emb, src_labels, src_preds, src_pen_emb = utils.get_embedding(self.model, self.src_train_loader, self.device, \
                                                                       self.num_classes, \
                                                                       self.args, with_emb=True, emb_dim=emb_dim)
        matrix = torch.cdist(src_pen_emb, tgt_pen_emb)
        matrix = matrix + 1.0
        matrix = 1.0 / matrix

        ind = torch.sort(matrix, descending=True, dim=0).indices
        ind_split = torch.split(ind, 1, dim=1)
        print("ind_split的长度是； " + str(len(ind_split)))
        print("target_dataloader的长度是: " + str(len(data_loader.sampler)))
        ind_split = [id.squeeze() for id in ind_split]
        vr_src = src_labels.unsqueeze(-1).repeat(1, len(self.src_train_loader.sampler))
        label_list = []
        for i in range(0, len(data_loader.sampler)):
            _row = ind_split[i].long()  # 第i个源域样本在目标域最近的编号  不对
            #  上面dim=0排序之后  应该是第i个目标域样本在源域中距离最近的样本编号
            _col = (torch.ones(len(self.src_train_loader.sampler)) * i).long()
            _val = vr_src[_row, _col]  # val出来的是 根据上面那些下标在
            print(_val.shape)
            top_n_val = _val[[j for j in range(0, 40)]]
            label_list.append(top_n_val)

            all_top_labels = torch.stack(label_list, dim=1)
        assigned_tgt_labels = torch.mode(all_top_labels, dim=0).values
        flat_src_labels = src_labels.squeeze()

        sim_matrix_split = torch.split(matrix, 1, dim=1)
        sim_matrix_split = [_id.squeeze() for _id in sim_matrix_split]

        simratio_score = []  # sim-ratio (knn conf measure) for all tgt

        for i in range(0, len(data_loader.sampler)):  # nln: nearest like neighbours, nun: nearest unlike neighbours
            t_label = assigned_tgt_labels[i]
            nln_mask = (flat_src_labels == t_label)

            nun_mask = ~(flat_src_labels == t_label)
            nun_sim_all = sim_matrix_split[i][nun_mask]

            len1 = len(nun_sim_all)
            nun_sim_r = torch.narrow(torch.sort(nun_sim_all, descending=True)[0], 0, 0, len(nun_sim_all))

            nln_sim_all = sim_matrix_split[i][nln_mask]
            nln_sim_r = torch.narrow(torch.sort(nln_sim_all, descending=True)[0], 0, 0, len(nln_sim_all))

            nln_sim_score = 1.0 * torch.sum(nln_sim_r)
            num_sim_score = torch.sum(nun_sim_r)

            conf_score = (nln_sim_score / num_sim_score).item()  # sim ratio : confidence score
            simratio_score.append((1.0 - conf_score))

        print(simratio_score)
        sort_ranking_score, ind_tgt = torch.sort(torch.tensor(simratio_score), descending=True)

        scores = (DivScore * sort_ranking_score).sort(descending=True)[1]
        # Sample from top-2 % instances, as recommended by authors
        # top_N = int(len(scores) * 0.02)
        # q_idxs = np.random.choice(scores[:top_N].cpu().numpy(), n, replace=False)

        # stat = sorted(stat, key=lambda x: x[1], reverse=True)
        # index = argsort(-DivScore)  # tensor类型
        q_idxs = []
        for i in range(0, n):
            q_idxs.append(scores[i].item())

        print("q_idxs的内容是: " + str(q_idxs))
        return q_idxs


@register_strategy('CLUE')
class CLUESampling(SamplingStrategy):
    """
    Implements CLUE: CLustering via Uncertainty-weighted Embeddings
    """

    #  def __init__(self, dset, tgt_loader, train_idx, model, device, args, balanced=False):
    def __init__(self, src_train_loader, dset, tgt_loader, train_idx, model,  discriminator ,device, src_seq_loader,args):
        super(CLUESampling, self).__init__(src_train_loader, dset, tgt_loader, train_idx, model, discriminator ,device, src_seq_loader,args)
        # Committee consistency hyperparameters
        self.randaug_n = 3  # RandAugment number of consecutive transformations
        self.randaug_m = 2.0  # RandAugment severity
        self.committee_size = 3  # Committee size

        # Pass in hyperparams to dataset
        self.tgt_loader.dataset.committee_size = self.committee_size
        self.tgt_loader.dataset.ra_obj.n = self.randaug_n
        self.tgt_loader.dataset.ra_obj.m = self.randaug_m

        # self.T = self.args.clue_softmax_t

        # train_loader打乱transform之后 每次取一批次 是否还是同一幅照片的不同形式

    def query(self, n):  # n是选取的
        print("train_idx长度: " + str(len(self.train_idx)))
        # self.idxs_lb = np.zeros(len(self.train_idx), dtype=bool) idxs_lb index labeled 初始全部赋值为0 因为初始时候是没有样本会被标注的

        # return np.random.choice(np.where(self.idxs_lb == 0)[0], n, replace=False)
        idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]  #
        train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])

        print("self.train_idx[idxs_unlabled]的长度: " + str(
            len(self.train_idx[idxs_unlabeled])))  # 这边实现采样是每次都要重新K-MEANS 选取样本
        data_loader = torch.utils.data.DataLoader(self.dset, sampler=train_sampler, num_workers=4, \
                                                  batch_size=self.args.batch_size, drop_last=False)
        print("data_loader中train_sampler的长度： " + str(len(data_loader.sampler)))  # 根据idxs_unlabeled选取样本

        if self.args.cnn == 'LeNet':
            emb_dim = 500
        elif self.args.cnn == 'ResNet50':
            emb_dim = 32
        device = self.device
        model = self.model
        model.eval()
        # Get embedding of target instances
        tgt_emb, tgt_lab, tgt_preds, tgt_pen_emb = utils.get_embedding(self.model, data_loader, self.device,\
                                                                       self.num_classes, \
                                                                       self.args, with_emb=True, emb_dim=emb_dim)
        # target instances' embedding shape
        tgt_pen_emb = tgt_pen_emb.cpu().numpy()
        tgt_scores = nn.Softmax(dim=1)(tgt_emb/2.0)
        tgt_scores += 1e-8
        sample_weights = -(tgt_scores * torch.log(tgt_scores)).sum(1).cpu().numpy()   # 用熵作权重

        # Run weighted K-means over embeddings
        km = KMeans(n)
        km.fit(tgt_pen_emb, sample_weight=sample_weights)

        # Find nearest neighbors to inferred centroids
        dists = euclidean_distances(km.cluster_centers_, tgt_pen_emb)

        sort_idxs = dists.argsort(axis=1)

        # q_idxs'change
        q_idxs = []
        ax, rem = 0, n # n是聚类个数
        while rem > 0:
            q_idxs.extend(list(sort_idxs[:, ax][:rem]))
            q_idxs = list(set(q_idxs))
            rem = n - len(q_idxs)
            ax += 1

        print("query函数中的idxs_unlabeled[q_idxs]:" + str(q_idxs))
        print("idxs_unlabeled[q_idxs]: " + str(idxs_unlabeled[q_idxs]))
        return idxs_unlabeled[q_idxs]