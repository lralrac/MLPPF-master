#!/usr/bin/env python
# coding:utf-8

import torch
import torch.nn as nn
from dataset_preprocessing.classification_dataset import ClassificationDataset as cDataset
from model.classification.classifier import Classifier
from model.embedding import PositionEmbedding
import torch.nn.functional as F

class Textcnn(Classifier):
    def __init__(self, dataset, config):
        super(Textcnn, self).__init__(dataset, config)

        self.pad = dataset.token_map[dataset.VOCAB_PADDING]

        seq_max_len = config.feature.max_token_len
        filter_sizes =  eval(config.Textcnn. filter_sizes)

        self.position_enc = PositionEmbedding(seq_max_len,
                                              config.embedding.dimension,
                                              self.pad)

        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.Textcnn.num_filters, (k, config.embedding.dimension)) for k in filter_sizes])

        hidden_size = config.embedding.dimension
        self.linear1 = torch.nn.Linear(config.Textcnn.num_filters * len(filter_sizes), hidden_size//2)
        self.linear2 = torch.nn.Linear(hidden_size//2, len(dataset.label_map))

        self.dropout = torch.nn.Dropout(p=config.train.hidden_layer_dropout)


    def update_lr(self, optimizer, epoch):
        if epoch > self.config.train.num_epochs_static_embedding:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = self.config.optimizer.learning_rate
        else:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = 0

    def forward(self, batch):


        src_seq = batch[cDataset.DOC_TOKEN].to(self.config.device)
        embedding = self.token_embedding(src_seq)


        # Prepare masks
        batch_lens = (src_seq != self.pad).sum(dim=-1)
        src_pos = torch.zeros_like(src_seq, dtype=torch.long)
        for row, length in enumerate(batch_lens):
            src_pos[row][:length] = torch.arange(1, length + 1)
        emb_output = embedding  +   self.position_enc(src_pos)
        # emb_output = embedding

        out = emb_output.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        return self.linear2(self.linear1(out))

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x