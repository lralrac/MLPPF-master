#!/usr/bin/env python
# coding:utf-8

from model.model_util import init_tensor
import torch

class SumAttention(torch.nn.Module):
    def __init__(self, input_dimension, attention_dimension, device, dropout=0):
        super(SumAttention, self).__init__()

        self.attention_matrix = \
            torch.nn.Linear(input_dimension, attention_dimension).to(device)
        self.attention_vector = torch.nn.Linear(attention_dimension, 1, bias=False).to(device)
        init_tensor(self.attention_matrix.weight)
        init_tensor(self.attention_vector.weight)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, inputs):
        if inputs.size(1) == 1:
            return self.dropout(inputs.squeeze())

        u = torch.tanh(self.attention_matrix(inputs))
        v = self.attention_vector(u)
        alpha = torch.nn.functional.softmax(v, 1).squeeze().unsqueeze(1)

        return self.dropout(torch.matmul(alpha, inputs).squeeze())

