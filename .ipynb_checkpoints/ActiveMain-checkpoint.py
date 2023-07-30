# -*- coding: utf-8 -*-
import sys
import os
import random
import distutils
from distutils import util
import argparse
from omegaconf import OmegaConf
import pprint
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import PLNLUtils
from PseudoLabelingUtil import pseudo_labeling

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)
torch.cuda.manual_seed(1234)
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from Task_net import ResNet50Fc
from datasets.base import ASDADataset
from sample import *


# from visualize import new_TSNE
def run_active_adaptation(args, source_model, src_dset, num_classes, device):
    """
    	Runs active domain adaptation experiments
    	"""

    # Load source data
    src_train_loader, src_val_loader, src_test_loader, src_train_idx, src_seq_loader = src_dset.get_loaders()

    # Load target data 需要k transformed versions
    # Setup target data loader
    # src_dset = ASDADataset(args.source, args.LDS_type, is_target=False, img_dir=args.img_dir,batch_size=args.batch_size)  # 数据集类的实例化
    target_dset = ASDADataset(args.target, is_target=True, img_dir=args.img_dir, batch_size=args.batch_size,
                              valid_ratio=0)  # 主动学习验证集的比率就为0
    target_train_dset, target_val_dset, target_test_dset = target_dset.get_dsets()  # k transformed versions datasets

    target_train_loader, target_val_loader, target_test_loader, target_train_idx, _ = target_dset.get_loaders()

    budget_all = len(target_train_idx)
    print(budget_all)
    args.total_budget = budget_all * 0.05
    print(args.total_budget)

    test_acc = 0
    ####################################################################################
    # 测试环节
    test_acc, cm_before = utils.test(source_model, device, target_test_loader, "test",
                                     num_classes)  # confusion matrix's function
    per_class_acc_before = cm_before.diagonal().numpy() / cm_before.sum(axis=1).numpy()
    per_class_acc_before = per_class_acc_before.mean() * 100

    out_str = '{}->{}, Before {}:\t Avg. acc={:.2f}%\tAgg. acc={:.2f}%'.format(args.source, args.target, args.da_strat,
                                                                               per_class_acc_before, test_acc)
    print(out_str)
    print('\n------------------------------------------------------\n')

    # Run unsupervised DA at round 0, where applicable
    discriminator = None
    if args.da_strat != 'mme':  # MME
        print('Round 0: Unsupervised DA to target via {}'.format(args.da_strat))
        model, src_model, discriminator = utils.run_unsupervised_da(source_model, src_train_loader, None,
                                                                    target_train_loader, \
                                                                    target_train_idx, num_classes, device, args)

        # Evaluate adapted source model on target test
        start_perf, _ = utils.test(model, device, target_test_loader, "test", num_classes)
        out_str = '{}->{} performance (After {}): {:.2f}'.format(args.source, args.target, args.da_strat, start_perf)
        print(out_str)
        print('\n------------------------------------------------------\n')
    else:
        model = source_model
        

    #################################################################
    # Setup
    #################################################################
    target_accs = defaultdict(list)  # 目前尚未采取相应的DA Strategy
    sampling_ratio = [(args.total_budget / args.num_rounds) * n for n in range(args.num_rounds + 1)]  # 采样比率存在的问题（）

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
        # print(sampling_ratio[1:])
        # print(len(sampling_ratio[1:]))  # len(sampling_ratio[1:]) 代表从第一个数开始索引 这边长度是30
        tqdm_rat = trange(len(sampling_ratio[1:]))  #
        target_accs[0.0].append(test_acc)  # start_perf 是test_acc

        # Making a copy for current
        curr_model = copy.deepcopy(model)
        curr_source_model = curr_model

        # Keep track of labeled vs unlabeled data. it's very important
        main_idxs_lb = np.zeros(len(target_train_idx), dtype=bool)
        idxs_lb = np.zeros(len(target_train_idx), dtype=bool)  # index of labeled

        # Instantiate active sampling strategy
        sampling_strategy = get_strategy(args.al_strat, src_train_loader, target_train_dset, target_train_loader,
                                         target_train_idx, \
                                         curr_model, discriminator, device, src_seq_loader,
                                         args)  # default: Clue-main  主动学习策略的实例化
        # target_aug_loader = sampling_strategy.tgt_loader # 类的实例化中作为RandAugument

        # tqdm_rat = trange(len(sampling_ratio[1:]))
        train_lbl_idx = []
        train_unlbl_idx = []
        best_model = None
        for ix in tqdm_rat:  # Iterate over Active DA rounds
            ratio = sampling_ratio[ix + 1]
            tqdm_rat.set_description('# Target labels={:d}'.format(int(ratio)))
            tqdm_rat.refresh()

            # Select instances via AL strategy
            print('\nSelecting instances...')
            idxs = sampling_strategy.query(int(sampling_ratio[1]))  # 每一轮应该选择的样本的序号 如果返回的是q_idxs
            # idxs_unlabeled = np.arange(len(target_train_idx))
            # idxs = idxs_u  nlabeled[idxs]
            idxs_lb[idxs] = True
            train_lbl_idx = idxs
            train_lbl_idx = []
            for i in range(0,len(target_train_loader.sampler)):
                if idxs_lb[i] == True:
                    train_lbl_idx.append(i)
            print("train_lbl_idx的长度： " + str(len(train_lbl_idx)))
            for i in range(0, len(target_train_idx)):
                if idxs_lb[i] == False:
                    train_unlbl_idx.append(i)
            sampling_strategy.update(idxs_lb)
            # Update model with new data via DA strategy
            best_model, optimizer = sampling_strategy.train(target_train_dset, da_round=(ix + 1), \
                                                            src_loader=src_train_loader, \
                                                            src_model=curr_source_model)

            # Evaluate on target test and train splits
            print("")
            test_perf, _ = utils.test(best_model, device, target_test_loader, "test", num_classes)
            train_perf, _ = utils.test(best_model, device, target_train_loader, "test", num_classes)
            best_model = best_model
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
    # parser.add_argument('--id', type=str, default='Derived1', help="Experiment identifier")
    parser.add_argument('--id', type=str, help="Experiment identifier")
    parser.add_argument('--model_init', type=str, default='source', help="Active DA model initialization")
    # Load existing configuration?
    parser.add_argument('--load_from_cfg', type=lambda x: bool(distutils.util.strtobool(x)), default=True,
                        help="Load from config?")
    parser.add_argument('--cfg_file', type=str, help="Experiment configuration file",
                        default="config/OfficeHome/Consistencyderived.yml")

    # Experimental details
    parser.add_argument('--runs', type=int, default=1, help="Number of experimental runs")
    parser.add_argument('--source', default="Clipart", help="Source dataset")
    parser.add_argument('--target', default="Art", help="Target dataset")
    # parser.add_argument('--img_dir', type=str, help="Data directory where images are stored", default="data/OfficeHome")
    parser.add_argument('--img_dir', type=str, help="Data directory where images are stored", default="data/VisDA2017")
    parser.add_argument('--total_budget', type=int, default=220, help="Total target budget")
    parser.add_argument('--num_rounds', type=int, default=3, help="Target dataset number of splits")  # 30 num_rounds

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
    src_dset = ASDADataset(args.source, is_target=False, img_dir=args.img_dir, batch_size=args.batch_size)  # 数据集类的实例化
    src_train_dset, src_val_dset, src_test_dset = src_dset.get_dsets()
    src_train_loader, src_val_loader, src_test_loader, train_idx, _ = src_dset.get_loaders()  # _代表train_idx

    num_classes = src_dset.get_num_classes()
    print('Number of classes: {}'.format(num_classes))

    ########################################################################################
    ######### train/setup a original model
    ########################################################################################
    # source_model:classifier conv_params....
    source_model = get_model(args.cnn, num_cls=num_classes, l2_normalize=args.l2_normalize,
                             temperature=args.temperature).to(device)

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

    # 测试环节
    print('Evaluating source checkpoint on {} test set...'.format(args.source))
    _, cm_source = utils.test(best_source_model, device, src_test_loader, split="test",
                              num_classes=num_classes)  # confusion martix's function
    per_class_acc_source = cm_source.diagonal().numpy() / cm_source.sum(axis=1).numpy()
    per_class_acc_source = per_class_acc_source.mean() * 100
    out_str = '{} Avg. acc.: {:.2f}% '.format(args.source, per_class_acc_source)
    print(out_str)

    # new_TSNE(best_source_model, src_train_loader, device, args, num_classes)

    # Run active adaptation experiments
    target_accs = run_active_adaptation(args, best_source_model, src_dset, num_classes, device)
    pp.pprint(target_accs)


if __name__ == '__main__':
    main()
