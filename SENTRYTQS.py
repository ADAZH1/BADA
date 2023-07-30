# -*- coding: utf-8 -*-
import sys
import os
import random
import distutils
from distutils import util
import argparse
from omegaconf import OmegaConf
import copy
import pprint
from collections import defaultdict
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm, trange

import numpy as np
import torch

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)
torch.cuda.manual_seed(1234)
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from adapt.models.models import get_model
import utils
from datasets.base import ASDADataset
from sample import *
# Compute classified results' variance/std
# 代码思路：先计算分类器对于单个dataloader的分类结果，然后计算传过来的5个dataloader 5个分类结果整个方差
# 上面思路感觉有点问题，这边的train_loader是一批数据里所有的样本 可能不能保证计算的时候 就比如A在loader1中出现次序是1 是否在后面每个loader都是1
# 重新参考TQS代码，思路貌似又没有问题。所以需要确保5个train_loader中的数据和标签是一致的，才能进行后续实验。
# 计算每个样本的得分需要参考TQS代码，最后加入候选样本集的部分需要参考CLUE代码，损失的话同样需要参考TQS代码
# 先写单个dataloader的分类结果（其实和传统的分类模型一致(test) ）
def get_uncertainty(fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5): # Paper's first chapter
    fc2_s = nn.Softmax(-1)(fc2_s)
    fc2_s2 = nn.Softmax(-1)(fc2_s2)
    fc2_s3 = nn.Softmax(-1)(fc2_s3)
    fc2_s4 = nn.Softmax(-1)(fc2_s4)
    fc2_s5 = nn.Softmax(-1)(fc2_s5)

    fc2_s = torch.unsqueeze(fc2_s, 1)
    fc2_s2 = torch.unsqueeze(fc2_s2, 1)
    fc2_s3 = torch.unsqueeze(fc2_s3, 1)
    fc2_s4 = torch.unsqueeze(fc2_s4, 1)
    fc2_s5 = torch.unsqueeze(fc2_s5, 1)
    print("fc2_s: "+fc2_s)
    c = torch.cat((fc2_s, fc2_s2, fc2_s3, fc2_s4, fc2_s5), dim=1)
    d = torch.std(c, 1)
    print("d: "+d)
    uncertainty = torch.mean(d, 1)
    return uncertainty

# Module2: 增广样本之间的一致性 样本一致性的度量 其实就是比较原数据和增广数据之间的一致性个数 返回的都是一个特定的概率比值
def get_consistency(output, output1, output2, output3, output4, output5):
    # tgt_preds = score_t_og.max(dim=1)[1].reshape(-1)  # target predictions
    tgt_preds = output.max(dim=1)[1].reshape(-1)
    tgt_preds1 = output1.max(dim=1)[1].reshape(-1)
    tgt_preds2 = output2.max(dim=1)[1].reshape(-1)
    tgt_preds3 = output3.max(dim=1)[1].reshape(-1)
    tgt_preds4 = output4.max(dim=1)[1].reshape(-1)
    tgt_preds5 = output5.max(dim=1)[1].reshape(-1)

    ConsistencyList = [tgt_preds1, tgt_preds2, tgt_preds3, tgt_preds4, tgt_preds5]
    ConsistencyValue = 0.000
    for i in ConsistencyList:
        if tgt_preds == i:
            ConsistencyValue+=1
    return (5 - ConsistencyValue)/5


# 主要是两个模块的损失函数 不能照抄SENTRY中的函数
def QueryStrategy(args, model, device, train_loader, train_loader1, train_loader2, train_loader3, train_loader4, train_loader5):
    model.eval()
    # train_loader's structure
    iters = zip(train_loader, train_loader1, train_loader2, train_loader3, train_loader4, train_loader5)

    with torch.no_grad():
        for batch_idx, ((data, target, _), (data1, target1, _), (data2, target2, _), (data3, target3, _),
                        (data4, target4, _), (data5, target5, _)) in enumerate(iters):
            data = data.to(device)
            data1 = data1.to(device)
            data2 = data2.to(device)
            data3 = data3.to(device)
            data4 = data4.to(device)
            data5 = data5.to(device)

            y = model(data)
            y1 = model(data1)
            y2 = model(data2)
            y3 = model(data3)
            y4 = model(data4)
            y5 = model(data5)
            # Compute variance between output1 ~ output5
            get_uncertainty(y1, y2, y3, y4, y5)

def run_active_adaptation(args, source_model, src_dset, num_classes, device):
    """
    	Runs active domain adaptation experiments
    	"""

    # Load source data
    src_train_loader, src_val_loader, src_test_loader, src_train_idx = src_dset.get_loaders()

    # Load target data 需要k transformed versions
    # Setup target data loader
    # src_dset = ASDADataset(args.source, args.LDS_type, is_target=False, img_dir=args.img_dir,batch_size=args.batch_size)  # 数据集类的实例化
    target_dset = ASDADataset(args.target, is_target=True, img_dir=args.img_dir, batch_size=args.batch_size, valid_ratio=0)  # 主动学习验证集的比率就为0
    target_train_dset, target_val_dset, target_test_dset, target_train_dset1, target_train_dset2, target_train_dset3, target_train_dset4, target_train_dset5\
        = target_dset.get_dsets()  # k transformed versions datasets

    print("target_train_dset: " + target_train_dset.data[0])
    print("target_train_dset1: " + target_train_dset1.data[0])
    print("target_train_dset2: " + target_train_dset2.data[0])

    target_train_loader, target_train_loader1, target_train_loader2, target_train_loader3, target_train_loader4, target_train_loader5, \
        target_val_loader, target_test_loader, target_train_idx = target_dset.get_loaders()

    # 5个train_loader的单独测试数据
    test_acc, cm_before = utils.test(source_model, device, target_train_loader, split="test",
                                       num_classes=num_classes)  # confusion matrix's function
    per_class_acc_before = cm_before.diagonal().numpy() / cm_before.sum(axis=1).numpy()
    per_class_acc_before = per_class_acc_before.mean() * 100

    out_str = '{}->{}, Before {}:\t Avg. acc={:.2f}%\tAgg. acc={:.2f}%'.format(args.source, args.target, args.da_strat, per_class_acc_before, test_acc)
    print("原始train_loader: " + out_str)

    #################################################################
    # Setup
    #################################################################
    target_accs = defaultdict(list)    # 目前尚未采取相应的DA Strategy
    sampling_ratio = [(args.total_budget / args.num_rounds) * n for n in range(args.num_rounds + 1)]  # 采样比率存在的问题（）
    model = source_model
    exp_name = '{}_{}_{}_{}_{}runs_{}rounds_{}budget'.format(args.id, args.model_init, args.al_strat, args.da_strat, \
                                                             args.runs, args.num_rounds, args.total_budget)
    #################################################################
    # Main Active DA loop
    #################################################################
    tqdm_run = trange(args.runs)
    for run in tqdm_run:
        tqdm_run.set_description('Run {}'.format(str(run)))
        tqdm_run.refresh()
        # sampling_ratio = [(args.total_budget/args.num_rounds) * n for n in range(args.num_rounds+1)]
        print(sampling_ratio[1:])
        print(len(sampling_ratio[1:]))  # len(sampling_ratio[1:]) 代表从第一个数开始索引 这边长度是30
        tqdm_rat = trange(len(sampling_ratio[1:]))  #
        target_accs[0.0].append(test_acc)  # start_perf 是test_acc

        # Making a copy for current run
        curr_model = copy.deepcopy(model)
        curr_source_model = curr_model

        # Keep track of labeled vs unlabeled data. it's very important
        idxs_lb = np.zeros(len(target_train_idx), dtype=bool)  # index of labeled

        # Instantiate active sampling strategy
        sampling_strategy = get_strategy(args.al_strat, target_train_dset, target_train_dset1, target_train_dset2, target_train_dset3, \
                                         target_train_dset4, target_train_dset5, target_train_idx,  \
                                         curr_model, device, args)  # default: Clue-main  主动学习策略的实例化
        # tqdm_rat = trange(len(sampling_ratio[1:]))
        for ix in tqdm_rat:  # Iterate over Active DA rounds
            ratio = sampling_ratio[ix + 1]
            tqdm_rat.set_description('# Target labels={:d}'.format(int(ratio)))
            tqdm_rat.refresh()

            # Select instances via AL strategy
            print('\nSelecting instances...')
            idxs = sampling_strategy.query(int(sampling_ratio[1]))  # 每一轮应该选择的样本的序号
            idxs_lb[idxs] = True
            sampling_strategy.update(idxs_lb)

            # Update model with new data via DA strategy
            best_model = sampling_strategy.train(target_train_dset, da_round=(ix + 1), \
                                                 src_loader=src_train_loader, \
                                                 src_model=curr_source_model)

            # Evaluate on target test and train splits
            test_perf, _ = utils.test(best_model, device, target_test_loader, num_classes)
            train_perf, _ = utils.test(best_model, device, target_train_loader, split='train')

            out_str = '{}->{} Test performance (Round {}, # Target labels={:d}): {:.2f}'.format(args.source,
                                                                                                args.target, ix,
                                                                                                int(ratio), test_perf)
            out_str += '\n\tTrain performance (Round {}, # Target labels={:d}): {:.2f}'.format(ix, int(ratio),
                                                                                               train_perf)
            print('\n------------------------------------------------------\n')
            print(out_str)

            target_accs[ratio].append(test_perf)

    # Log at the end of every run
    wargs = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
    target_accs['args'] = wargs
    utils.log(target_accs, exp_name)

    return target_accs


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Example')
    # Experiment identifiers
    parser.add_argument('--id', type=str, default='debug', help="Experiment identifier")
    parser.add_argument('--model_init', type=str, default='source', help="Active DA model initialization")
    # Load existing configuration?
    parser.add_argument('--load_from_cfg', type=lambda x: bool(distutils.util.strtobool(x)), default=True, help="Load from config?")
    parser.add_argument('--cfg_file', type=str, help="Experiment configuration file", default="config/OfficeHome/sentryderived.yml")

    # Experimental details
    parser.add_argument('--runs', type=int, default=1, help="Number of experimental runs")
    parser.add_argument('--source', default="Art", help="Source dataset")
    parser.add_argument('--target', default="Product", help="Target dataset")
    parser.add_argument('--img_dir', type=str, help="Data directory where images are stored", default="data/OfficeHome")
    parser.add_argument('--total_budget', type=int, default=300, help="Total target budget")
    parser.add_argument('--num_rounds', type=int, default=30, help="Target dataset number of splits")  # 30 num_rounds

    parser.add_argument('--adapt_lr', default=0.0002, help="adapt_lr")

    args_cmd = parser.parse_args()

    # use_cuda = args_cmd.gpu and torch.cuda.is_available()
    if args_cmd.load_from_cfg:  # 处理配置文件
        args_cfg = dict(OmegaConf.load(args_cmd.cfg_file))  # make configuration into dictionary
        args_cmd = vars(args_cmd)  # make args_cmd into vars'style
        for k in args_cmd.keys():
            if args_cmd[k] is not None: args_cfg[k] = args_cmd[k]
        args = OmegaConf.create(args_cfg)
    else:
        args = args_cmd

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(args)

    device = torch.device("cuda") if args.use_cuda else torch.device("cpu")

    ########################################################################################
    #########  setup source data loaders
    ########################################################################################
    print('Loading {} dataset'.format(args.source))
    src_dset = ASDADataset(args.source, is_target=False, img_dir=args.img_dir, batch_size=args.batch_size) # 数据集类的实例化
    src_train_dset, src_val_dset, src_test_dset = src_dset.get_dsets()
    src_train_loader, src_val_loader, src_test_loader, train_idx = src_dset.get_loaders()  # _代表train_idx
    num_classes = src_dset.get_num_classes()
    print('Number of classes: {}'.format(num_classes))
    print("source_train_loader:" + src_train_loader.__str__())

    ########################################################################################
    ######### train/setup a original model
    ########################################################################################
    # source_model:classifier conv_params....
    source_model = get_model(args.cnn, num_cls=num_classes, l2_normalize=args.l2_normalize,
                             temperature=args.temperature)

    source_file = '{}_{}_source.pth'.format(args.source, args.cnn)
    source_path = os.path.join('checkpoints', 'source', source_file)

    if os.path.exists(source_path):  # Load existing source model
        print('Loading source checkpoint: {}'.format(source_path))
        source_model.load_state_dict(torch.load(source_path, map_location=device), strict=False)
        best_source_model = source_model
    else:
        print('\nSource checkpoint not found, training...')
        best_source_model = utils.train_source_model(source_model, src_train_loader, src_val_loader, num_classes, args,
                                                     device)  # resnet source-only code

    print('Evaluating source checkpoint on {} test set...'.format(args.source))
    _, cm_source = utils.test(best_source_model, device, src_test_loader, split="test",
                              num_classes=num_classes)  # confusion martix's function
    per_class_acc_source = cm_source.diagonal().numpy() / cm_source.sum(axis=1).numpy()
    per_class_acc_source = per_class_acc_source.mean() * 100
    out_str = '{} Avg. acc.: {:.2f}% '.format(args.source, per_class_acc_source)
    print(out_str)

    # Run active adaptation experiments
    target_accs = run_active_adaptation(args, best_source_model, src_dset, num_classes, device)
    pp.pprint(target_accs)


if __name__ == '__main__':
    main()
