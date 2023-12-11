#!/usr/bin/env python
# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
class CNNLayer(nn.Module):
    def __init__(self, d_model, kernel_size):
        super(CNNLayer, self).__init__()
        self.w_1 = nn.Conv1d(d_model, d_model, kernel_size, padding=1)  # position-wise

    def forward(self, x):
            residual = x
            output = x.transpose(1, 2)
            output = F.relu(self.w_1(output))
            output = output.transpose(1, 2)
            output =output + residual
            return output


