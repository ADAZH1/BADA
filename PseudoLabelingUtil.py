import random
import time
import pickle
from collections import Counter
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from misc import AverageMeter, accuracy
import sys
import os
import random

def pseudo_labeling(args,  device, data_loader, model, itr, unc, log_prob, num_classes, PLacc,ix):
    # unc log_prob就是前面传过来的很重要的数据
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    pseudo_idx = []
    pseudo_target = []

    gt_target = []
    idx_list = []
    gt_list = []
    target_list = []
    nl_mask = []

    model.eval()
    if ix ==0:
      PLTHE = 0.6
    if ix ==1:
      PLTHE = 0.7
    if ix ==2:
      PLTHE = 0.8
    if ix ==3:
      PLTHE = 0.9
    if ix ==4:
      PLTHE = 1.0
    #PLTHE = 0.5
    print(PLTHE)
    print("pseudo-label generation and selection")

    # print(log_prob.shape)   # log_prob存放着 之前计算每个样本的方差?

    example = [101,102]   # 范例索引出log_prob中相对应下标的数据
    log_example = log_prob[example]
    # print(log_example)
    # print(log_example.shape)

    # all_scores按照下标存放了所有的分数 下面需要按照每一轮indexs取出相应的分数 并且建立成相关矩阵 像原代码的那种64*10
    # print(len(data_loader.sampler))
    # log_prob *= 10
    with torch.no_grad():
        # 此处的data_loader是之前传过来的unlbl_loader 需要分配伪标签
        # score代表得分    data_loader必须要有不同增广下的概率
        for batch_idx, (( inputs, data_t_raug), targets, indexs, _) in enumerate(data_loader):
            # 下面就是对unlbl_loader中的数据进行处理 得分策略！   indexs存疑
            # print(indexs)  # 这边输出的index是顺序的 很奇怪
            data_time.update(time.time() - end)
            inputs = inputs.to(device)
            targets = targets.to(device)  # batch_size
            outputs = model(inputs)

            out_prob = []
            out_prob_nl = []
            out_prob.append(F.softmax(outputs, dim=1))
            out_prob_nl.append(F.softmax(outputs / args.temp_nl, dim=1))
            for data_t_aug_curr in data_t_raug:  # 遍历每个数据的增广结果
                outputs = model(data_t_aug_curr.to(device))
                out_prob.append(F.softmax(outputs, dim=1))  # for selecting positive pseudo-labels
                out_prob_nl.append(F.softmax(outputs, dim=1))  # for selecting negative pseudo-labels

            out_prob = torch.stack(out_prob)
            out_prob_nl = torch.stack(out_prob_nl)
            # print("out_prob的形状： " + str(out_prob.shape) + " out_prob_nl的形状： " + str(out_prob_nl.shape))  # [10, 64, 10]

            out_std = torch.std(out_prob, dim=0)  # 按照[10, 64, 10] 第一个维度 就是10那个维度（10个增广过的）
            out_std_nl = torch.std(out_prob_nl, dim=0)  # [64, 10]
            # 每个样本10*10 最后成为1*10 类似2*4的矩阵 按照第0维求解其标准差 参考范例
            # print("out_std的形状： " + str(out_std.shape) + " out_std_nl的形状： " + str(out_std_nl.shape))   #  这十个的标准差？ 64*10 64个样本 每个样本十个结果 每个结果都是计算出来的方差

            out_prob = torch.mean(out_prob, dim=0)
            # print("out_prob : "+ str(out_prob))
            out_prob_nl = torch.mean(out_prob_nl, dim=0)  # [64, 10]
            # print("out_prob的形状： " + str(out_prob.shape) + " out_prob_nl的形状： " + str(out_prob_nl.shape))  # 同理 标准差

            max_value, max_idx = torch.max(out_prob, dim=1)  # 目前的out_prob就是64*10   返回在那个标签上的下标  返回最大概率的所在下标
            # print("max_value的形状: " + str(max_value.shape) + " max_idx的形状： " + str(max_idx.shape))  # 最大的那个概率作为伪标签   max_idx
            max_std = out_std.gather(1, max_idx.view(-1, 1))  # out_std就是64*10  按照上面求出的max_idx出来的下标
            # print("max_std的形状： " + str(max_std.shape)) # sampler * 1 就是概率最大的那个值（伪标签） 标准差


            # log_prob是一些数据  目标是组成类似out_std的矩阵
            # out_prob = log_prob[indexs]  #  out_prob是 32 * 65  按照indexs索引取出来
            # unconfidence = unc[indexs]
            # print(out_prob.shape)

            # diversity
            # 其实可以直接根据index得到（样本得分策略保存这些信息）
            # selected_idx = (max_value >= args.tau_p) * (max_std.squeeze(1) < args.kappa_p)
            # print(out_prob)
            # print("out_std_nl: " + str(out_std_nl))
            #print("out_prob_nl"+str(out_prob_nl))
            interm_nl_mask = ((out_std_nl < 0.007 ) * (out_prob_nl < 1)) * 1    # Core Code  选择负学习的相关标签 我这边是只根据
            
            # print(interm_nl_mask)
            # out_std_nl 是 64*10数据链条(其实可以从其他函数传进来  具体就是根据index索引构造出这样的数组)
            # interm_nl_mask = ((out_std_nl < args.kappa_n) * (out_prob_nl.cpu().numpy() < args.tau_n)) * 1  # Core Code
            # 原作者是 将out_std_nl作为一个评价标准 另外就是out_prob_nl这个其实就是输出概率 非常传统的概率输出行为 所以原作者是用概率标准差和概率输出作为评价行为
            # 现在将其改编为 方差（其实和标准差类似）小于一个阈值 还有就是当前概率的熵小于一个阈值
            # 10.6改编的目前还是基于方差来做

            # print(interm_nl_mask.shape) # 128*10
            # print(interm_nl_mask)

            # max_value, max_idx = torch.max(outputs, dim=1)  # 目前的out_prob就是64*10   返回在那个标签上的下标  返回最大概率的所在下标

            #  manually setting the argmax value to zero
            for enum, item in enumerate(max_idx.cpu().numpy()):
                interm_nl_mask[enum, item] = 0  # 概率最大的那个标签置为0了 就不可能是补标签 其余的再经过上一轮inter_nl_mask的选择（更为精细地补标签选择）
            nl_mask.extend(interm_nl_mask.cpu().numpy().tolist())  # negative learning
            # print(nl_mask)
            # indexs
            idx_list.extend(indexs.numpy().tolist())  # 这一个batch 64个样本存下来
            gt_list.extend(targets.cpu().numpy().tolist())  # 原来数据集样本的真实标签
            # print(gt_list)
            target_list.extend(max_idx.cpu().numpy().tolist())  # 预测概率最大的样本标签
            # print(target_list)

            #  selecting positive pseudo-labels
            # interm_nl_mask = ((out_std_nl < args.kappa_n) * (out_prob_nl.cpu().numpy() < args.tau_n)) * 1
            # print(outputs)
            # print(max_value)
            # print(max_value.shape)
            # print(max_std.squeeze(1).shape)
            # selected_idx = (max_value >= 0.7) * ( max_std.squeeze(1) < args.kappa_p)  # 得到的是索引好的下标？ 应该就是64*1?
            # print(max_value)
            #print("max_std.squeeze(1) " + str(max_std.squeeze(1)))
            # selected_idx = (max_value >= 0.6) * (max_std.squeeze(1) < 0.5)
            # 关于正标签概率的补充实验  0.5 0.6 0.7 0.8 0.9
            selected_idx = (max_value >= 0.7) * (max_std.squeeze(1) < 0.4)
            
            # print(selected_idx)
            # 选择正样本 就是置信度比较大的样本 就是128   满足这些条件的样本是可以给伪标签的

            # print("selected_idx")
            # print(selected_idx)
            #  正学习  [selected_idx].cpu().numpy()
            # print("selected_idx的形状： " + str(selected_idx.shape))
            # print(max_std.squeeze(1)[selected_idx])

            # print(" pseudo_target的长度: " + str(len(pseudo_target)))
            pseudo_target.extend(max_idx[selected_idx].cpu().numpy().tolist())  # selected_idx选择出的是进行正学习的样本标号（True False标出）
            # max_idx
            # print(max_idx[selected_idx])
            # print(" pseudo_target的长度: " + str(len(pseudo_target)))

            pseudo_idx.extend(indexs[selected_idx].numpy().tolist())  # 这些被选择的样本下标被记录 表明已经选择成为伪标签
            # print(indexs[selected_idx])
            gt_target.extend(targets[selected_idx].cpu().numpy().tolist()) # 真实的标签？ 用处应该不大

            # loss = F.cross_entropy(outputs, targets.to(dtype=torch.long))
            # prec1, prec5 = accuracy(outputs[selected_idx], targets[selected_idx], topk=(1, 5))

            # losses.update(loss.item(), inputs.shape[0])
            # top1.update(prec1.item(), inputs.shape[0])
            # top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
        #     if not args.no_progress:
        #         data_loader.set_description(
        #             "Pseudo-Labeling Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
        #                 batch=batch_idx + 1,
        #                 iter=len(data_loader),
        #                 data=data_time.avg,
        #                 bt=batch_time.avg,
        #                 loss=losses.avg,
        #                 top1=top1.avg,
        #                 top5=top5.avg,
        #             ))
        # if not args.no_progress:
        #     data_loader.close()

    pseudo_target = np.array(pseudo_target) # 真的是伪标签？
    gt_target = np.array(gt_target)

    pseudo_idx = np.array(pseudo_idx)  # 被选择成为伪标签的样本下标

    # class balance the selected pseudo-labels
    # if itr < args.class_blnc - 1:
    #     min_count = 5000000  # arbitary large value
    #     for class_idx in range(args.num_classes):
    #         class_len = len(np.where(pseudo_target == class_idx)[0])
    #         if class_len < min_count:
    #             min_count = class_len
    #     min_count = max(25,
    #                     min_count)  # this 25 is used to avoid degenarate cases when the minimum count for a certain class is very low
    #
    #     blnc_idx_list = []
    #     for class_idx in range(args.num_classes):
    #         current_class_idx = np.where(pseudo_target == class_idx)
    #         if len(np.where(pseudo_target == class_idx)[0]) > 0:
    #
    #             sorted_maxstd_idx = np.argsort(current_class_maxstd)
    #             current_class_idx = current_class_idx[0][
    #                 sorted_maxstd_idx[:min_count]]  # select the samples with lowest uncertainty
    #             blnc_idx_list.extend(current_class_idx)
    #
    #     blnc_idx_list = np.array(blnc_idx_list)
    #     pseudo_target = pseudo_target[blnc_idx_list]
    #     pseudo_idx = pseudo_idx[blnc_idx_list]
    #     gt_target = gt_target[blnc_idx_list]

    pseudo_labeling_acc = (pseudo_target == gt_target) * 1
    #print(pseudo_labeling_acc)
    PLacc[itr].append(pseudo_labeling_acc)
    pseudo_labeling_acc = (sum(pseudo_labeling_acc) / len(pseudo_labeling_acc)) * 100
    print(f'Pseudo-Labeling Accuracy (positive): {pseudo_labeling_acc}, Total Selected: {len(pseudo_idx)}')

    pseudo_nl_mask = []
    pseudo_nl_idx = []
    nl_gt_list = []

    # print("idx_list:")
    # print(idx_list)
    num = 0
    for i in range(len(idx_list)):
        # print(sum(nl_mask[i]))
        if idx_list[i] not in pseudo_idx and sum(nl_mask[i]) > 0: # 大于0代表就是负学习中 样本存在些负标签   sum(nl_mask[i])>0 代表存在着补标签的选择
            num += 1
            pseudo_nl_mask.append(nl_mask[i])
            pseudo_nl_idx.append(idx_list[i])
            nl_gt_list.append(gt_list[i])
    # print(num)
    #print("伪标签生成中 nl_mask：")
    # print(nl_mask)
    nl_gt_list = np.array(nl_gt_list).astype('int64')
    pseudo_nl_mask = np.array(pseudo_nl_mask)
    # print(pseudo_nl_mask.shape)
    one_hot_targets = np.eye(num_classes)[nl_gt_list]
    one_hot_targets = one_hot_targets - 1
    one_hot_targets = np.abs(one_hot_targets)
    flat_pseudo_nl_mask = pseudo_nl_mask.reshape(1, -1)[0]
    #print(flat_pseudo_nl_mask.shape)  # pseudo_nl_mask.reshape(1, -1)[0] 的具体含义
    flat_one_hot_targets = one_hot_targets.reshape(1, -1)[0]
    flat_one_hot_targets = flat_one_hot_targets[np.where(flat_pseudo_nl_mask == 1)]
    flat_pseudo_nl_mask = flat_pseudo_nl_mask[np.where(flat_pseudo_nl_mask == 1)]

    nl_accuracy = (flat_pseudo_nl_mask == flat_one_hot_targets) * 1
    # nl_accuracy_final = (sum(nl_accuracy) / len(nl_accuracy)) * 100
    # print(
    #     f'Pseudo-Labeling Accuracy (negative): {nl_accuracy_final}, Total Selected: {len(nl_accuracy)}, Unique Samples: {len(pseudo_nl_mask)}')
    print(
        f'Pseudo-Labeling Accuracy (negative):      , Total Selected: {len(nl_accuracy)}, Unique Samples: {len(pseudo_nl_mask)}')
    pseudo_label_dict = {'pseudo_idx': pseudo_idx.tolist(), 'pseudo_target': pseudo_target.tolist(),
                         'nl_idx': pseudo_nl_idx, 'nl_mask': pseudo_nl_mask.tolist()}
    if pseudo_idx is None:
        print("pseudo_idx is None")
    if pseudo_target is None:
        print("pseudo_target is None")
    if pseudo_nl_idx is None:
        print("pseudo_nl_idx is None")
    if pseudo_nl_mask is None:
        print("pseudo_nl_mask is None")

    print(type(pseudo_label_dict))
    if pseudo_label_dict is None:
        print("pseudo_lbl_dict is None")
    return pseudo_label_dict