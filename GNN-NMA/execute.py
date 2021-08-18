# -*- encoding: utf-8 -*-
"""
@File    : execute.py
@Time    : 1/25/21 3:00 PM
@Author  : Liangliang
@Software: PyCharm
由于NVIDIA GeForce RTX 2080Ti的内存仅为11G,无法应对大规模的图神经网络训练,因此本算法只写了CPU版本
"""
import numpy as np
import networkx as nx
import time
from tqdm import tqdm
import torch
from Augment import Augment
from GNN_NMA import Net, train
from metric import metric


class ArgumentParser():#设置算法的参数
    def __init__(self, r=3, L=3, d=64, k=3, tau=2,lamda =2,gamma=2,eta=1,mu=2,epoch=500,lr=0.1, num=20 ,neurons_gnn=[300,250,200], neurons_mlp=[110,80,50], augmentation=[0,1,1,0]):
        self.r = r #阶数
        self.L = L #GNN隐含层层数
        self.d = d #数据的维数
        self.k = k #社团的数目
        self.tau = tau
        self.lamda = lamda
        self.gamma = gamma
        self.eta = eta
        self.mu = mu
        self.epoch = epoch #神经网络训练次数
        self.lr = lr #优化算法的学习率
        self.num = num #算法的运行次数
        self.neurons_gnn = neurons_gnn #GNN各层的神经元数目
        self.neurons_mlp = neurons_mlp #MLP各层的神经元数目
        self.augmentation = augmentation #共4位,每一位为0或1, 第1位代表是否使用边增广,第2位代表否使用噪声属性增广,第3位代表是否使用自动编码器对属性增广,第4位代表否使用联合增广


if __name__ == '__main__':
    parser = ArgumentParser()
    name = 'cora' #数据集名称
    edges = np.loadtxt('./data/'+name+'/edges.txt',dtype=np.int).tolist()
    labels = np.loadtxt('./data/'+name+'/labels.txt',dtype=np.int)
    flag = int(input('Is the input graph an attribute graph?(1: Yes. 2: No.) Please input:'))
    if flag == 1:
        features = np.loadtxt('./data/'+name+'/features.txt',dtype=np.float)
        features = torch.FloatTensor(features)
    else:
        features = sparse.identity(len(labels))
        features = features.tocoo()
        features = torch.sparse.FloatTensor(torch.LongTensor([features.row.tolist(), features.col.tolist()]), torch.FloatTensor(features.data))
    graph = nx.Graph()
    graph.add_nodes_from([i for i in range(len(labels))])
    graph.add_edges_from(edges)
    parser.d = int(features.shape[1])
    g = Augment(graph, features, parser)
    ACC = []
    R = []
    J = []
    ARI = []
    FM = []
    F1 = []
    Hubert = []
    K = []
    RT = []
    CD = []
    SS = []
    NMI = []
    Num = 20
    for _ in tqdm(range(Num)):
        net = Net([graph, g[0][0], g[1][0]], parser)
        pred = train(net, graph, parser, [features, g[0][1], g[1][1]])
        pred = np.argmax(pred, axis=1)
        AACC, RR, JJ, AARI, FFM, FF1, HHubert, KK, RRT, CCD, SSS, NNMI = metric(labels, pred)
        ACC.append(AACC)
        R.append(RR)
        J.append(JJ)
        ARI.append(AARI)
        F1.append(FF1)
        FM.append(FFM)
        Hubert.append(HHubert)
        K.append(KK)
        RT.append(RRT)
        CD.append(CCD)
        SS.append(SSS)
        NMI.append(NNMI)
    end = time.time()
    print('The data is {} and the results of SDCN are as follow:'.format(name))
    print('ACC={}$\pm${}'.format(round(np.mean(ACC), 4), round(np.std(ACC), 4)))
    print('R={}$\pm${}'.format(round(np.mean(R), 4), round(np.std(R), 4)))
    print('J={}$\pm${}'.format(round(np.mean(J), 4), round(np.std(J), 4)))
    print('ARI={}$\pm${}'.format(round(np.mean(ARI), 4), round(np.std(ARI), 4)))
    print('FM={}$\pm${}'.format(round(np.mean(FM), 4), round(np.std(FM), 4)))
    print('F1={}$\pm${}'.format(round(np.mean(F1), 4), round(np.std(F1), 4)))
    print('Hubert={}$\pm${}'.format(round(np.mean(Hubert), 4), round(np.std(Hubert), 4)))
    print('K={}$\pm${}'.format(round(np.mean(K), 4), round(np.std(K), 4)))
    print('RT={}$\pm${}'.format(round(np.mean(RT), 4), round(np.std(RT), 4)))
    print('CD={}$\pm${}'.format(round(np.mean(CD), 4), round(np.std(CD), 4)))
    print('SS={}$\pm${}'.format(round(np.mean(SS), 4), round(np.std(SS), 4)))
    print('NMI={}$\pm${}'.format(round(np.mean(NMI), 4), round(np.std(NMI), 4)))
    print('The time cost is', round((end - start) / Num, 4))