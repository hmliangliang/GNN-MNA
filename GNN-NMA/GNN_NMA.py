# -*- encoding: utf-8 -*-
"""
@File    : GNN-NMA.py
@Time    : 1/25/21 2:58 PM
@Author  : Liangliang
@Software: PyCharm
"""
import torch
import torch.nn as nn
import networkx as nx
import torch.optim as optim
import math
from torch.nn import functional as F
from MINE import Mutual_information
import copy

class Perceptron(nn.Module):
    def __init__(self, input_feat, parser):
        super(Perceptron,self).__init__()
        self.layer1 = nn.Linear(input_feat, parser.neurons_mlp[0])
        self.layer2 = nn.Linear(parser.neurons_mlp[0], parser.neurons_mlp[1])
        self.layer = nn.Linear(parser.neurons_mlp[1], parser.k)
    def forward(self, X):
        h = self.layer1(X)
        h = torch.tanh(h)
        h = self.layer2(h)
        h = torch.tanh(h)
        h = self.layer(h)
        h = F.normalize(h, p=2, dim=1)
        return h

class GNN(nn.Module):
    def __init__(self, parser):
        super(GNN, self).__init__()
        self.parser = parser
        self.linear1 = nn.Linear(parser.d, parser.neurons_gnn[0])
        self.attn_fc1 = nn.Linear(2*parser.neurons_gnn[0], 1)
        self.linear2 = nn.Linear(parser.neurons_gnn[0], parser.neurons_gnn[1])
        self.attn_fc2 = nn.Linear(2*parser.neurons_gnn[1], 1)
        self.linear3 = nn.Linear(parser.neurons_gnn[1], parser.neurons_gnn[2])

    def attention(self, graph, h, attn_fc):
        parser = self.parser
        N = graph.number_of_nodes()
        # 计算注意力
        a = {}
        for i in range(N):
            temp = []
            for j in list(graph.neighbors(i)):
                temp.append(float(torch.exp(F.leaky_relu(attn_fc(torch.cat((h[i, :], h[j, :]), dim=0)))).detach()))
            temp = [1 / sum(temp) * uu for uu in temp]
            uu=-1
            for j in list(graph.neighbors(i)):
                uu=uu+1
                a[str(i) + str(j)] = temp[int(uu)]
        del temp
        '''A = torch.zeros(N,N)
        for i in range(N):
            for j in range(N):
                A[i,j] = torch.dot(h[i,:],h[j,:])/(torch.norm(h[i,:])*torch.norm(h[j,:]))
        if parse.L > 1:
            temp = copy.deepcopy(A)
            for _ in range(parse.L-1):
                A = torch.mm(A, temp)
            del temp
        for i in range(N):
            for j in range(N):
                if float(A[i,j]) <= math.pow(0.6, parser.L):
                    A[i,j] = 0
        A = torch.softmax(A, dim=1)
        h1 = linear(h)'''
        h1 = copy.deepcopy(h.detach())
        for i in range(N):
            for j in range(N):
                Aij = float(torch.dot(h[i, :], h[j, :]) / (torch.norm(h[i, :]) * torch.norm(h[j, :])).detach())
                if Aij <= math.pow(0.6, parser.L):
                    Aij = 0
                b = torch.exp(Aij*F.leaky_relu(attn_fc(torch.cat((h[i, :], h[j, :]), dim=0))))
                h1[i,:] = h1[i,:] + b*h1[j,:]
                if j in list(graph.neighbors(i)):
                    h[i, :] = h[i, :] + a[str(i) + str(j)]*h[j, :]
        h = 0.5*(h+h1)
        return h

    def forward(self, graph, X):
        h = self.linear1(X)
        h = self.attention(graph, h, self.attn_fc1)
        h = F.relu(h)
        h = self.linear2(h)
        h = self.attention(graph, h, self.attn_fc2)
        h = F.relu(h)
        h = self.linear3(h)
        h = F.normalize(h, p=2, dim=1)
        return h


class Onlinenet(nn.Module):
    def __init__(self, parser, graph):
        super(Onlinenet, self).__init__()
        self.parser = parser
        self.graph = graph
        self.gnn = GNN(self.parser)
        self.project = Perceptron(parser.neurons_gnn[2], parser)
    def forward(self, X):
        h = self.gnn.forward(self.graph, X)
        h = self.project.forward(h)
        return h

class Targetnet(nn.Module):
    def __init__(self, parser, graph):
        super(Targetnet,self).__init__()
        self.parser = parser
        self.graph = graph
        self.gnn = GNN(parser)
        self.project = Perceptron(parser.neurons_gnn[2], parser)
    def forward(self, X):
        h = self.gnn.forward(self.graph, X)
        h = self.project.forward(h)
        return h

class Net(nn.Module):
    def __init__(self, graph, parser):#graph=[原始图,增广图,增广图,增广图,增广图]
        super(Net,self).__init__()
        self.graph = graph
        self.parser = parser
        self.Targetnet = Targetnet(parser, graph[0])
        self.Onlinenet1 = Onlinenet(parser, graph[1])
        self.Onlinenet2 = Onlinenet(parser, graph[2])
        #self.Onlinenet3 = Onlinenet(parser, graph[3])
    def forward(self, X):#X=[原始图节点特征,增广图节点特征,增广图节点特征,增广图节点特征,增广图节点特征]
        h = self.Targetnet.forward(X[0])
        h1 = self.Onlinenet1.forward(X[1])
        h2 = self.Onlinenet2.forward(X[2])
        #h3 = self.Onlinenet3.forward(X[3])
        return h, h1, h2

def Loss(parser, graph, h, h1, h2, X):
    print('It is computing loss!')
    L = torch.FloatTensor([len(list(graph.neighbors(i))) for i in range(graph.number_of_nodes())])
    Lmi = Mutual_information(h, h1) + Mutual_information(h, h2)
    Linfo = torch.norm(torch.mm(h, h.transpose(0,1))-torch.mm(X, X.transpose(0,1))) +  torch.norm(torch.mm(h1, h1.transpose(0,1))-torch.mm(X, X.transpose(0,1))) + torch.norm(torch.mm(h2, h2.transpose(0,1))-torch.mm(X, X.transpose(0,1)))
    Linfo = 1/n*Linfo
    Linfo = Linfo + X.shape[0]*torch.trace(torch.mm(h.transpose(0,1),torch.mm(L,h))) + X.shape[0]*torch.trace(torch.mm(h1.transpose(0,1),torch.mm(L,h1))) +  X.shape[0]*torch.trace(torch.mm(h2.transpose(0,1),torch.mm(L,h2)))
    Ls = torch.norm(h,'nuc') + torch.norm(h1,'nuc') + torch.norm(h2,'nuc')
    Lsim = torch.norm(h-h1) + torch.norm(h-h2)  + torch.norm(torch.mean(h, dim=0)-torch.mean(h1, dim=0)) + torch.norm(torch.mean(h, dim=0)-torch.mean(h2, dim=0))
    Lsim = 1/X.shape[0]*Lsim
    loss = Linfo + parser.lamda*Ls + parser.gamma*Lsim - parser.eta*Lmi
    return loss

def Momentum_uupdate(net, parser):#动量更新网络
    for param_k, param_q1, param_q2 in zip(net.Targetnet.parameters(), net.Onlinenet1.parameters(), net.Onlinenet2.parameters(), net.Onlinenet3.parameters()):
        param_q.data = 0.25*(param_q1.data + param_q2.data)
        param_k.data = param_k.data*parser.mu + param_q.data*(1. - parser.mu)
    return net

def train(net, graph, parser, X):
    h_best = []
    n_max = 50
    n = 0
    loss_best = 2**31-1
    optimizer = optim.SGD(list(net.Onlinenet1.parameters()) + list(net.Onlinenet2.parameters()), lr = parser.lr, momentum=0.9)
    for epoch in range(parser.epoch):
        print('epoch=',epoch)
        print('Begain to train network!')
        h, h1, h2 = net.forward(X)
        print('Network finshes forward!')
        loss = Loss(parser, graph, h, h1, h2, X[0])
        print('epoch: {}    loss:{}'.format(epoch, loss))
        if loss < loss_best:
            loss_best = loss
            h_best = h
            n = 0
        else:
            n = n + 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        net = Momentum_uupdate(net)
        if n >= n_max:
            print('Early stop!')
            break
    return F.softmax(h_best, dim=1).detach().numpy()
