# -*- coding: utf-8 -*-
import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .solver import register_solver

sys.path.append('../../')
import utils

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)
torch.cuda.manual_seed(1234)


class BaseSolver:
    """
	Base DA solver class
	"""

    def __init__(self, net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args):
        self.net = net
        self.src_loader = src_loader
        self.tgt_sup_loader = tgt_sup_loader
        self.tgt_unsup_loader = tgt_unsup_loader
        # aug_loader
        self.train_idx = np.array(train_idx)
        self.tgt_opt = tgt_opt
        self.da_round = da_round
        self.device = device
        self.args = args
        self.current_step = 0
        self.param_lr_c = []
        for param_group in self.tgt_opt.param_groups:
            self.param_lr_c.append(param_group["lr"])

    def solve(self, epoch):
        pass


@register_solver('ft')
class TargetFTSolver(BaseSolver):
    """
    Finetune on target labels
    """

    def __init__(self, net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args):
        super(TargetFTSolver, self).__init__(net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt,
                                             da_round, device, args)

    # 目标域中必定是给定了一些标签，有监督的话，效果是会提升 这是毋庸置疑的
    def solve(self, epoch):
        """
        Finetune on target labels
        """

        self.net.train()
        if (self.da_round > 0): tgt_sup_iter = iter(self.tgt_sup_loader)

        # test_acc, cm_before = utils.test(self.net, self.device, self.tgt_unsup_loader, split="test",
        #                                  num_classes=65)  # confusion matrix's function
        # per_class_acc_before = cm_before.diagonal().numpy() / cm_before.sum(axis=1).numpy()
        # per_class_acc_before = per_class_acc_before.mean() * 100
        #
        # out_str = '{}->{}, Before {}:\t Avg. acc={:.2f}%\tAgg. acc={:.2f}%'.format(self.args.source, self.args.target,
        #                                                                            self.args.da_strat,
        #                                                                            per_class_acc_before,
        #                                                                            test_acc)
        # print("solve中 self model target_train_loader: " + out_str)

        info_str = '[Train target finetuning] Epoch: {}'.format(epoch)
        while True:
            try:
                self.current_step += 1
                (data_t, _), target_t, index = next(tgt_sup_iter)

                data_t, target_t = data_t.to(self.device), target_t.to(self.device)
            except:
                break
            self.tgt_opt.zero_grad()

            output = self.net(data_t)
            loss = nn.CrossEntropyLoss()(output, target_t)

            # info_str = '[Train target finetuning] Epoch: {}'.format(epoch)
            # info_str += ' Target Sup. Loss: {:.3f}'.format(loss.item())

            loss.backward()
            self.tgt_opt.step()

        # test_acc, cm_before = utils.test(self.net, self.device, self.tgt_unsup_loader, split="test",
        #                                  num_classes=65)  # confusion matrix's function
        # per_class_acc_before = cm_before.diagonal().numpy() / cm_before.sum(axis=1).numpy()
        # per_class_acc_before = per_class_acc_before.mean() * 100
        #
        # out_str = '{}->{}, Before {}:\t Avg. acc={:.2f}%\tAgg. acc={:.2f}%'.format(self.args.source, self.args.target,
        #                                                                            self.args.da_strat,
        #                                                                            per_class_acc_before,
        #                                                                            test_acc)
        # print("solve中一次训练过后 self model target_train_loader: " + out_str)
        # if epoch % 10 == 0: print(info_str)


@register_solver('dann')
class DANNSolver(BaseSolver):
    """
    Implements DANN from Unsupervised Domain Adaptation by Backpropagation: https://arxiv.org/abs/1409.7495
    """

    #  def __init__(self, net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args):
    def __init__(self, net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args):
        super(DANNSolver, self).__init__(net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt,
                                         da_round, device, args)

    def solve(self, epoch, disc, disc_opt):
        """
        Semisupervised adaptation via DANN: XE on labeled source + XE on labeled target + \
                                    ent. minimization on target + DANN on source<->target
        """

        gan_criterion = nn.CrossEntropyLoss()
        cent = utils.ConditionalEntropyLoss().to(self.device)

        self.net.train()
        disc.train()

        if self.da_round == 0:
            src_sup_wt, lambda_unsup, lambda_cent = 1.0, 0.1, 0.01  # Hardcoded for unsupervised DA
        else:
            src_sup_wt, lambda_unsup, lambda_cent = self.args.src_sup_wt, self.args.unsup_wt, self.args.cent_wt
            tgt_sup_iter = iter(self.tgt_sup_loader)
        joint_loader = zip(self.src_loader, self.tgt_unsup_loader)

        for batch_idx, (((data_s, _, _), label_s, _), ((data_tu, _, _), label_tu, _)) in enumerate(joint_loader):

            data_s, label_s = data_s.to(self.device), label_s.to(self.device)

            data_tu = data_tu.to(self.device)
            if self.da_round > 0:
                try:
                    (data_ts, _, _), label_ts, _ = next(tgt_sup_iter)
                    data_ts, label_ts = data_ts.to(self.device), label_ts.to(self.device)
                except:
                    break

            # zero gradients for optimizers

            self.tgt_opt.zero_grad()
            disc_opt.zero_grad()
            # Train with target labels
            score_s = self.net(data_s)
            xeloss_src = src_sup_wt * nn.CrossEntropyLoss()(score_s, label_s)


            info_str = "[Train DANN] Epoch: {}".format(epoch)
            info_str += " Src Sup loss: {:.3f}".format(xeloss_src.item())
            xeloss_tgt = 0
            if self.da_round > 0:
                score_ts = self.net(data_ts)
                xeloss_tgt = nn.CrossEntropyLoss()(score_ts, label_ts)
                info_str += " Tgt Sup loss: {:.3f}".format(xeloss_tgt.item())

            # extract and concat features
            score_tu = self.net(data_tu)
            f = torch.cat((score_s, score_tu), 0)

            # predict with discriminator
            f_rev = utils.ReverseLayerF.apply(f)
            pred_concat = disc(f_rev)

            target_dom_s = torch.ones(len(data_s)).long().to(self.device)
            target_dom_t = torch.zeros(len(data_tu)).long().to(self.device)
            label_concat = torch.cat((target_dom_s, target_dom_t), 0)

            # compute loss for disciminator
            loss_domain = gan_criterion(pred_concat, label_concat)
            loss_cent = cent(score_tu)

            loss_final = (xeloss_src + xeloss_tgt) + (lambda_unsup * loss_domain) + (lambda_cent * loss_cent)

            loss_final.backward()

            self.tgt_opt.step()
            disc_opt.step()

            # log net update info
            info_str += " DANN loss: {:.3f}".format(lambda_unsup * loss_domain.item())
            info_str += " Ent Loss: {:.3f}".format(lambda_cent * loss_cent.item())

        if epoch % 10 == 0: print(info_str)


@register_solver('consistency')
class ConsistencySolver(BaseSolver):
    """"
    在MME基础上加上KL散度作为第一阶段的总损失,所以需要其他转换形式的数据集？参照SENTRY论文
    """
    # def __init__(self, net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, ):


@register_solver('mme')
class MMESolver(BaseSolver):
    """
	Implements MME from Semi-supervised Domain Adaptation via Minimax Entropy: https://arxiv.org/abs/1904.06487
	"""

    #  def __init__(self, net, src_loader, tgt_sup_loader, tgt_unsup_loader, target_aug_loader, train_idx, tgt_opt, da_round, device, args):
    def __init__(self, net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args):
        super(MMESolver, self).__init__(net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args)


    def solve(self, epoch):
        """
		Semisupervised adaptation via MME: XE on labeled source + XE on labeled target + \
										adversarial ent. minimization on unlabeled target
		"""

        self.net.train()
        src_sup_wt, lambda_adent, kl_wt = self.args.src_sup_wt, self.args.unsup_wt, self.args.kl_wt

        if self.da_round == 0:
            src_sup_wt, lambda_unsup = 1.0, 0.1
        else:
            src_sup_wt, lambda_unsup = self.args.src_sup_wt, self.args.unsup_wt
            tgt_sup_iter = iter(self.tgt_sup_loader)

        joint_loader = zip(self.src_loader, self.tgt_unsup_loader)
        # for batch_idx, ((_, data_t_og, data_t_raug), label_t, indices_t) in enumerate(tqdm(self.tgt_loader)):
        # for batch_idx, ((data, _, _), target, _) in enumerate(tqdm(train_loader)):
        # for batch_idx, (((data_s, _), label_s, _), ((data_tu, _), label_tu, _)) in enumerate(joint_loader):
        # print(self.tgt_aug_loader.dataset.committee_size)
        for batch_idx, (((data_s, _, _), label_s, _), ((data_tu, _, data_aug_list), label_tu, _)) in enumerate(
                joint_loader):
            data_s, label_s = data_s.to(self.device), label_s.to(self.device)
            data_tu = data_tu.to(self.device)

            if self.da_round > 0:
                try:
                    # data_ts, label_ts = next(tgt_sup_iter)
                    # (data, _, _), target, _
                    (data_ts, _, _), label_ts, _ = next(tgt_sup_iter)
                    data_ts, label_ts = data_ts.to(self.device), label_ts.to(self.device)
                except:
                    break

            # zero gradients for optimizer
            self.tgt_opt.zero_grad()

            # log basic adapt train info
            info_str = "[Train Minimax Entropy] Epoch: {}".format(epoch)

            # extract features
            score_s = self.net(data_s)  # src_loss
            xeloss_src = src_sup_wt * nn.CrossEntropyLoss()(score_s, label_s)

            # log discriminator update info
            info_str += " Src Sup loss: {:.3f}".format(xeloss_src.item())

            xeloss_tgt = 0
            if self.da_round > 0:
                score_ts = self.net(data_ts)
                xeloss_tgt = nn.CrossEntropyLoss()(score_ts, label_ts)
                info_str += " Tgt Sup loss: {:.3f}".format(xeloss_tgt.item())

            # # Consistency Loss Use KLDivergence
            # KLLoss = 0
            # if self.da_round > 0:
            #     # KLLoss = 0
            #     score_tu = self.net(data_tu)
            #     for data_aug_curr in data_aug_list:
            #         score_t_aug_curr = self.net(data_aug_curr.to(self.device))
            #         KLLoss = KLLoss + F.kl_div(score_t_aug_curr.softmax(dim=-1).log(), score_tu.softmax(dim=-1), reduction='mean')
            #     KLLoss = kl_wt *KLLoss
            #     info_str += " Tgt Aug KLLoss: {:.3f}".format(KLLoss.item())

            #
            xeloss = xeloss_src + xeloss_tgt
            xeloss.backward()
            self.tgt_opt.step()

        if epoch % 10 == 0: print(info_str)

        
@register_solver('Tradmme')
class TradMMESolver(BaseSolver):
    """
    Implements MME from Semi-supervised Domain Adaptation via Minimax Entropy: https://arxiv.org/abs/1904.06487
    """

    #  def __init__(self, net, src_loader, tgt_sup_loader, tgt_unsup_loader, target_aug_loader, train_idx, tgt_opt, da_round, device, args):
    def __init__(self, net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args):
        super(TradMMESolver, self).__init__(net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args)


    def solve(self, epoch):
        """
            Semisupervised adaptation via MME: XE on labeled source + XE on labeled target + \
                                            adversarial ent. minimization on unlabeled target
        """

        self.net.train()
        src_sup_wt, lambda_adent, kl_wt = self.args.src_sup_wt, self.args.unsup_wt, self.args.kl_wt

        if self.da_round == 0:
            src_sup_wt, lambda_unsup = 1.0, 0.1
        else:
            src_sup_wt, lambda_unsup = self.args.src_sup_wt, self.args.unsup_wt
            tgt_sup_iter = iter(self.tgt_sup_loader)

        joint_loader = zip(self.src_loader, self.tgt_unsup_loader)
        # for batch_idx, ((_, data_t_og, data_t_raug), label_t, indices_t) in enumerate(tqdm(self.tgt_loader)):
        # for batch_idx, ((data, _, _), target, _) in enumerate(tqdm(train_loader)):
        # for batch_idx, (((data_s, _), label_s, _), ((data_tu, _), label_tu, _)) in enumerate(joint_loader):
        # print(self.tgt_aug_loader.dataset.committee_size)
        for batch_idx, (((data_s, _, _), label_s, _), ((data_tu, _, data_aug_list), label_tu, _)) in enumerate(
                joint_loader):
            data_s, label_s = data_s.to(self.device), label_s.to(self.device)
            data_tu = data_tu.to(self.device)

            if self.da_round > 0:
                try:
                    # data_ts, label_ts = next(tgt_sup_iter)
                    # (data, _, _), target, _
                    (data_ts, _, _), label_ts, _ = next(tgt_sup_iter)
                    data_ts, label_ts = data_ts.to(self.device), label_ts.to(self.device)
                except:
                    break

            # zero gradients for optimizer
            self.tgt_opt.zero_grad()
            
            # log basic adapt train info
            info_str = "[Train Minimax Entropy] Epoch: {}".format(epoch)

            # extract features
            score_s = self.net(data_s)  # src_loss
            xeloss_src = src_sup_wt * nn.CrossEntropyLoss()(score_s, label_s)

            # log discriminator update info
            info_str += " Src Sup loss: {:.3f}".format(xeloss_src.item())

            xeloss_tgt = 0
            if self.da_round > 0:
                score_ts = self.net(data_ts)
                xeloss_tgt = nn.CrossEntropyLoss()(score_ts, label_ts)
                info_str += " Tgt Sup loss: {:.3f}".format(xeloss_tgt.item())

            # # Consistency Loss Use KLDivergence
            # KLLoss = 0
            # if self.da_round > 0:
            #     # KLLoss = 0
            #     score_tu = self.net(data_tu)
            #     for data_aug_curr in data_aug_list:
            #         score_t_aug_curr = self.net(data_aug_curr.to(self.device))
            #         KLLoss = KLLoss + F.kl_div(score_t_aug_curr.softmax(dim=-1).log(), score_tu.softmax(dim=-1), reduction='mean')
            #     KLLoss = kl_wt *KLLoss
            #     info_str += " Tgt Aug KLLoss: {:.3f}".format(KLLoss.item())

            #
            xeloss = xeloss_src + xeloss_tgt
            xeloss.backward()
            
            self.tgt_opt.step()
            
            # Add adversarial entropy
            self.tgt_opt.zero_grad()

            score_tu = self.net(data_tu, reverse_grad=True)
            probs_tu = F.softmax(score_tu, dim=1)
            loss_adent = lambda_adent * torch.mean(torch.sum(probs_tu * (torch.log(probs_tu + 1e-5)), 1))
            loss_adent.backward()

            self.tgt_opt.step()

        if epoch % 10 == 0: print(info_str)
