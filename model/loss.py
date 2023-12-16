#!/usr/bin/env python
# coding:utf-8

import torch
import torch.nn as nn
from util import Type


class LossType(Type):
    """Standard names for loss type
    """
    SOFTMAX_CROSS_ENTROPY = "SoftmaxCrossEntropy"
    SOFTMAX_FOCAL_CROSS_ENTROPY = "SoftmaxFocalCrossEntropy"
    SIGMOID_FOCAL_CROSS_ENTROPY = "SigmoidFocalCrossEntropy"
    BCE_WITH_LOGITS = "BCEWithLogitsLoss"

    @classmethod
    def str(cls):
        return ",".join([cls.SOFTMAX_CROSS_ENTROPY,
                         cls.SOFTMAX_FOCAL_CROSS_ENTROPY,
                         cls.SIGMOID_FOCAL_CROSS_ENTROPY,
                         cls.BCE_WITH_LOGITS])


class ActivationType(Type):
    """Standard names for activation type
    """
    SOFTMAX = "Softmax"
    SIGMOID = "Sigmoid"

    @classmethod
    def str(cls):
        return ",".join([cls.SOFTMAX,
                         cls.SIGMOID])


class FocalLoss(nn.Module):
    def __init__(self, label_size, activation_type=ActivationType.SOFTMAX,
                 gamma=2.0, alpha=0.25, epsilon=1.e-9):
        super(FocalLoss, self).__init__()
        self.num_cls = label_size
        self.activation_type = activation_type
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, logits, target):
        """
        Args:
            logits: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        if self.activation_type == ActivationType.SOFTMAX:
            idx = target.view(-1, 1).long()
            one_hot_key = torch.zeros(idx.size(0), self.num_cls,
                                      dtype=torch.float,
                                      device=idx.device)
            one_hot_key = one_hot_key.scatter_(1, idx, 1)
            logits = torch.softmax(logits, dim=-1)
            loss = -self.alpha * one_hot_key * \
                   torch.pow((1 - logits), self.gamma) * \
                   (logits + self.epsilon).log()
            loss = loss.sum(1)
        elif self.activation_type == ActivationType.SIGMOID:
            multi_hot_key = target
            logits = torch.sigmoid(logits)
            zero_hot_key = 1 - multi_hot_key
            loss = -self.alpha * multi_hot_key * \
                   torch.pow((1 - logits), self.gamma) * \
                   (logits + self.epsilon).log()
            loss += -(1 - self.alpha) * zero_hot_key * \
                    torch.pow(logits, self.gamma) * \
                    (1 - logits + self.epsilon).log()
        else:
            raise TypeError("Unknown activation type: " + self.activation_type
                            + "Supported activation types: " +
                            ActivationType.str())
        return loss.mean()

class ClassificationLoss(torch.nn.Module):
    def __init__(self, label_size, class_weight=None,
                 loss_type=LossType.SOFTMAX_CROSS_ENTROPY):
        super(ClassificationLoss, self).__init__()
        self.label_size = label_size
        self.loss_type = loss_type
        if loss_type == LossType.SOFTMAX_CROSS_ENTROPY:
            self.criterion = torch.nn.CrossEntropyLoss(class_weight)
        elif loss_type == LossType.SOFTMAX_FOCAL_CROSS_ENTROPY:
            self.criterion = FocalLoss(label_size, ActivationType.SOFTMAX)
        elif loss_type == LossType.SIGMOID_FOCAL_CROSS_ENTROPY:
            self.criterion = FocalLoss(label_size, ActivationType.SIGMOID)
        elif loss_type == LossType.BCE_WITH_LOGITS:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            raise TypeError(
                "Unsupported loss type: %s. Supported loss type is: %s" % (
                    loss_type, LossType.str()))

    def forward(self, logits, target,
                use_hierar=False,
                is_multi=False,
                *argvs):
        device = logits.device
        # if use_hierar:
        #     assert self.loss_type in [LossType.BCE_WITH_LOGITS,
        #                               LossType.SIGMOID_FOCAL_CROSS_ENTROPY]
        #     if not is_multi:
        #         target = torch.eye(self.label_size)[target].to(device)
        #     hierar_penalty, hierar_paras, hierar_relations = argvs[0:3]
        #     return self.criterion(logits, target) + \
        #            hierar_penalty * self.cal_recursive_regularize(hierar_paras,
        #                                                           hierar_relations,
        #                                                           device)
        # else:
        if is_multi:
                assert self.loss_type in [LossType.BCE_WITH_LOGITS,
                                          LossType.SIGMOID_FOCAL_CROSS_ENTROPY]
        else:
                if self.loss_type not in [LossType.SOFTMAX_CROSS_ENTROPY,
                                          LossType.SOFTMAX_FOCAL_CROSS_ENTROPY]:
                    target = torch.eye(self.label_size)[target].to(device)
        return self.criterion(logits, target)

    def cal_recursive_regularize(self, paras, hierar_relations, device="cpu"):
        """ Only support hierarchical text classification with BCELoss
        """
        recursive_loss = 0.0
        for i in range(len(paras)):
            if i not in hierar_relations:
                continue
            children_ids = hierar_relations[i]
            if not children_ids:
                continue
            children_ids_list = torch.tensor(children_ids, dtype=torch.long).to(
                device)
            children_paras = torch.index_select(paras, 0, children_ids_list)
            parent_para = torch.index_select(paras, 0,
                                             torch.tensor(i).to(device))
            parent_para = parent_para.repeat(children_ids_list.size()[0], 1)
            diff_paras = parent_para - children_paras
            diff_paras = diff_paras.view(diff_paras.size()[0], -1)
            recursive_loss += 1.0 / 2 * torch.norm(diff_paras, p=2) ** 2
        return recursive_loss
