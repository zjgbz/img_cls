# -*- coding: utf-8 -*-
"""
# @file name  : my_loss.py
# @author     : https://github.com/TingsongYu
# @date       : 2021-02-28 10:08:00
# @brief      : 新的loss
"""
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)  # log_p 向量
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)

        # index_tensor = target.unsqueeze(-1)
        # index_array = index_tensor.numpy()
        # index = torch.LongTensor(index_array)
        index = target.unsqueeze(-1)
        
        weight.scatter_(-1, index, (1. - self.smoothing))  # Q向量
        loss = (-weight * log_prob).sum(dim=-1).mean()  # log_p * Q 再相加
        return loss


if __name__ == '__main__':

    output = torch.tensor([[4.0, 5.0, 10.0], [1.0, 5.0, 4.0], [1.0, 15.0, 4.0]])
    label = torch.tensor([2, 1, 1], dtype=torch.int64)

    criterion = LabelSmoothLoss(0.01)
    loss = criterion(output, label)

    print("CrossEntropy:{}".format(loss))