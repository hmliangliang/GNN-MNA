# -*- encoding: utf-8 -*-
"""
@File    : AE.py
@Time    : 1/27/21 2:51 PM
@Author  : Liangliang
@Software: PyCharm
"""
import numpy as np
import torch
import torch.nn as nn
from torch import optim as optimizer


class autoencoder(nn.Module):
    def __init__(self, X, number=[200, 100, 50]):
        super(autoencoder, self).__init__()
        self.layer1 = nn.Linear(number[0], number[1])
        self.layer2 = nn.Linear(number[1],number[2])
        self.layer3 = nn.Linear(number[2], number[1])
        self.layer4 = nn.Linear(number[1], number[0])
        self.X = torch.FloatTensor(X)
    def forward(self):
        h = self.layer1(self.X)
        h = torch.sigmoid(h)
        h = self.layer2(h)
        h = torch.sigmoid(h)
        h = self.layer3(h)
        h = torch.sigmoid(h)
        h = self.layer4(h)
        return h

def Loss(X, Y):
    loss = torch.norm(X-Y, 'fro')/X.shape[1]
    return loss

def train(X):
    X = torch.FloatTensor(X)
    number = [X.shape[1], 100, 50]
    net = autoencoder(X, number)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.152)
    for _ in range(100):
        Y = net.forward()
        loss = Loss(X, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return Y.detach()

