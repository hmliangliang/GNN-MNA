# -*- encoding: utf-8 -*-
"""
@File    : MINE.py
@Time    : 1/30/21 1:42 PM
@Author  : Liangliang
@Software: PyCharm
"""
'''
参考代码 https://blog.csdn.net/xyisv/article/details/111354132
参考文献: Belghazi M I, Baratin A, Rajeshwar S, et al. Mutual information neural estimation. International Conference on Machine Learning. 2018: 531-540.
'''
import torch
import torch.nn as nn
import numpy as np


class MINE(nn.Module):
    def __init__(self, data_dim=5, hidden_size=10):
        super(MINE, self).__init__()
        self.layers = nn.Sequential(nn.Linear(2 * data_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))
    def forward(self, x, y):
        batch_size = x.size(0)
        tiled_x = torch.cat([x, x, ], dim=0)
        idx = torch.randperm(batch_size)

        shuffled_y = y[idx]
        concat_y = torch.cat([y, shuffled_y], dim=0)
        inputs = torch.cat([tiled_x, concat_y], dim=1)
        logits = self.layers(inputs)

        pred_xy = logits[:batch_size]
        pred_x_y = logits[batch_size:]
        loss = - np.log2(np.exp(1)) * (torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y))))
        # compute loss, you'd better scale exp to bit
        return loss


def Mutual_information(X,Y):
    hidden_size = 200
    n_epoch = 450
    data_dim = X.shape[1]
    model = MINE(data_dim, hidden_size).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    plot_loss = []
    all_mi = []
    for epoch in range(n_epoch):
        x_sample = X.cuda()
        y_sample = Y.cuda()
        loss = model(x_sample, y_sample)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        all_mi.append(-loss.item())
    return sum(all_mi[-100:])/100
