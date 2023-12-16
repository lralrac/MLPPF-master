#!/usr/bin/env python
# coding:utf-8

import numpy as np
import torch
import torch.nn as nn

from model.model_util import ActivationType
from model.model_util import FAN_MODE
from model.model_util import InitType
from model.model_util import init_tensor
from util import Logger
from util import Type, ModeType


class EmbeddingType(Type):
    """Standard names for embedding type
    The following keys are defined:
    * `EMBEDDING`: Return the embedding after lookup.
    * `REGION_EMBEDDING`: Return the region embedding.
        Reference: A New Method of Region Embedding for Text Classification
    """
    EMBEDDING = 'embedding'
    REGION_EMBEDDING = 'region_embedding'
    
    @classmethod
    def str(cls):
        return ",".join([cls.EMBEDDING, cls.REGION_EMBEDDING])


class EmbeddingProcessType(Type):
    """Standard names for embedding mode
    Given the vocab tensor shape[batch_size, sequence_len].
    The following keys are defined:
    * `FLAT`: Normal mode, return tensor shape will be
    *         [batch_size, sequence_len, embedding_size]
    * `MEAN`: Mean mode, return tensor shape will be
    *         [batch_size, embedding_size]
    * `SUM`: Sum mode, return tensor shape will be
    *        [batch_size, embedding_size]
    """
    FLAT = 'flat'
    MEAN = 'mean'
    SUM = 'sum'
    
    @classmethod
    def str(cls):
        return ",".join([cls.FLAT, cls.MEAN, cls.SUM])


class Embedding(torch.nn.Module):
    def __init__(self, dict_map, embedding_dim, name, config, padding_idx=None,
                 pretrained_embedding_file=None, mode=EmbeddingProcessType.FLAT,
                 dropout=0, init_type=InitType.XAVIER_UNIFORM, low=0, high=1,
                 mean=0, std=1, activation_type=ActivationType.NONE,
                 fan_mode=FAN_MODE.FAN_IN, negative_slope=0,
                 model_mode=ModeType.TRAIN):
        super(Embedding, self).__init__()
        self.logger = Logger(config)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.mode = mode
        if self.mode == EmbeddingProcessType.FLAT:
            self.embedding = torch.nn.Embedding(
                len(dict_map), embedding_dim, padding_idx=padding_idx)
        else:
            self.embedding = torch.nn.EmbeddingBag(
                len(dict_map), embedding_dim, mode=mode)
        #     Randomly initialize k-mer embeddings
        embedding_lookup_table = init_tensor(
            tensor=torch.empty(len(dict_map), embedding_dim),
            init_type=init_type, low=low, high=high, mean=mean, std=std,
            activation_type=activation_type, fan_mode=fan_mode,
            negative_slope=negative_slope)
        model_mode=ModeType.TRAIN
        if model_mode == ModeType.TRAIN and \
                pretrained_embedding_file is not None and \
                pretrained_embedding_file != "":
            self.load_pretrained_embedding(
                embedding_lookup_table, dict_map, embedding_dim, name,
                pretrained_embedding_file)
        if padding_idx is not None:
            embedding_lookup_table[padding_idx] = 0.0
        self.embedding.weight.data.copy_(embedding_lookup_table)
        self.embedding.data_dic=dict_map

    def forward(self, vocab_ids, offset=None):
        if self.mode == EmbeddingProcessType.FLAT:
            embedding = self.embedding(vocab_ids)
        else:
            embedding = self.embedding(vocab_ids, offset)
        return self.dropout(embedding)

    def load_pretrained_embedding(
            self, embedding_lookup_table, dict_map, embedding_dim, name,
            pretrained_embedding_file):
        self.logger.warn(
            "Load %s embedding from %s" % (name, pretrained_embedding_file))
        with open(pretrained_embedding_file) as fin:
            num_pretrained = 0
            for line in fin:
                data = line.strip().split(',')
                # Check embedding info
                if len(data) == 2:
                    assert int(data[1]) == embedding_dim, \
                        "Pretrained embedding dim not matching: %s, %d" % (
                            data[1], embedding_dim)
                    continue
                if data[0] not in dict_map:
                    continue
                embedding = torch.FloatTensor([float(i) for i in data[1:]])
                embedding_lookup_table[dict_map[data[0]]] = embedding
                num_pretrained += 1
        self.logger.warn(
            "Total dict size of %s is %d" % (name, len(dict_map)))
        self.logger.warn("Size of pretrained %s embedding is %d" % (
            name, num_pretrained))
        self.logger.warn(
            "Size of randomly initialize %s embedding is %d" % (
                name, len(dict_map) - num_pretrained))

class PositionEmbedding(torch.nn.Module):
    ''' Reference: attention is all you need '''

    def __init__(self, seq_max_len, embedding_dim, padding_idx):
        super(PositionEmbedding, self).__init__()

        self.position_enc = nn.Embedding.from_pretrained(
            self.get_sinusoid_encoding_table(seq_max_len + 1,
                                             embedding_dim,
                                             padding_idx=padding_idx),
            freeze=True)

    def forward(self, src_pos):

        return self.position_enc(src_pos)

    @staticmethod
    def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):

        def cal_angle(position, hid_idx):

            return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

        sinusoid_table = np.array(
            [get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        if padding_idx is not None:
            # zero vector for padding dimension
            sinusoid_table[padding_idx] = 0.

        return torch.FloatTensor(sinusoid_table)
