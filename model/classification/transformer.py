#!/usr/bin/env python
# coding:utf-8

import torch
import torch.nn as nn
from dataset_preprocessing.classification_dataset import ClassificationDataset as cDataset
from model.classification.classifier import Classifier
from model.embedding import PositionEmbedding
import torch.nn.functional as F
import copy
class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out

class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out

class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
        Return:
            The tensor after self-attention and the attention tensor
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context

class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.3):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  # Scaling factor
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  # Residual connection
        out = self.layer_norm(out)
        return out


class Transformer(Classifier):
    def __init__(self, dataset, config):
        super(Transformer, self).__init__(dataset, config)

        self.pad = dataset.token_map[dataset.VOCAB_PADDING]

        seq_max_len = config.feature.max_token_len
        hidden_size = config.embedding.dimension
        hidden =  config.Transformer.hidden
        self.position_enc = PositionEmbedding(seq_max_len,
                                              config.embedding.dimension,
                                              self.pad)
        self.encoder = Encoder(config.Transformer.dim_model, config.Transformer.num_head, hidden, config.Transformer.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.Transformer.num_encoder)])
        self.linear1 = torch.nn.Linear(hidden_size, hidden_size//2)
        self.linear2 = torch.nn.Linear(hidden_size//2, len(dataset.label_map))
        self.dropout = torch.nn.Dropout(p=config.train.hidden_layer_dropout)

    def update_lr(self, optimizer, epoch):
        if epoch > self.config.train.num_epochs_static_embedding:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = self.config.optimizer.learning_rate
        else:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = 0.0

    def forward(self, batch):
        src_seq = batch[cDataset.DOC_TOKEN].to(self.config.device)
        embedding = self.token_embedding(src_seq)
        # Prepare masks
        batch_lens = (src_seq != self.pad).sum(dim=-1)
        src_pos = torch.zeros_like(src_seq, dtype=torch.long)
        for row, length in enumerate(batch_lens):
            src_pos[row][:length] = torch.arange(1, length + 1)
        emb_output = embedding
        # emb_output = embedding + self.position_enc(src_pos)
        z=t=0

        for encoder in self.encoders:
            out = encoder(emb_output)
            if t!=0:
                ture_out=torch.cat((ture_out,out), dim=1)
            else:
                ture_out = out
            t=t+1
        out=torch.mean(ture_out,1)
        out=self.linear1(out)
        return self.linear2(out)
