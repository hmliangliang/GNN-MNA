# -*- encoding: utf-8 -*-
"""
@File    : Augment.py
@Time    : 1/25/21 2:59 PM
@Author  : Liangliang
@Software: PyCharm
"""
import numpy as np
import networkx as nx
import math
import copy
import torch
from AE import train

def Augment(graph, X, parser):
    augmentation = parser.augmentation
    X = X.numpy()
    result = []
    g = nx.Graph()
    #edge augmentation
    if augmentation[0] == 1:
        A = nx.adjacency_matrix(graph).todense()
        data = copy.deepcopy(A)
        g = copy.deepcopy(graph)
        for _ in range(parser.r - 1):
            data = np.dot(data, A)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i,j] == 1:#delete an edge
                    pij = len(list(nx.common_neighbors(graph,i,j)))/(len(list(list(g.neighbors(i))))+len(list(g.neighbors(j)))+2-len(list(nx.common_neighbors(graph,i,j)))) + np.dot(X[i,:],X[j,:])/(np.linalg.norm(X[i,:])*np.linalg.norm(X[j,:]))
                    pij = pij / 2
                    pij = pij + 1 / (1 + math.exp(-(len(list(list(g.neighbors(i)))) + len(list(g.neighbors(j))) + 2) / parser.tau))
                    pij = math.exp(-pij / 2) / (1 + math.exp(-pij/2))
                    p = np.random.rand()
                    if p <= pij:
                        g.remove_edge(i,j)
                        A[i,j] = 0
                        A[j, i] = 0
                else:#add an edge
                    pij = (data[i,j]/np.sum(data[i,:]) + np.dot(X[i, :], X[j, :]) / (np.linalg.norm(X[i, :]) * np.linalg.norm(X[j, :])))/2
                    pij = 1/(2+math.exp(-(1-pij)))
                    p = np.random.rand()
                    if p <= pij:
                        g.add_edge(i,j)
                        A[i,j] = 1
                        A[j, i] = 1
        del data
        result.append([g, torch.FloatTensor(X)])
    #attribute augmentation--noise
    if augmentation[1] == 1:
        noise = X + np.random.normal(loc=0.0, scale=1, size=(X.shape[0],X.shape[1]))
        noise = (noise- np.min(noise)) /(np.max(noise) - np.min(noise))
        result.append([graph, torch.FloatTensor(noise)])
    #attribute augmentation--auencoder
    if augmentation[2] == 1:
        Y = train(X)
        result.append([graph, torch.FloatTensor(Y)])
    #Union augmentation
    if augmentation[3] == 1:
        A = nx.adjacency_matrix(graph).todense()
        if augmentation[0] == 1:
            data = copy.deepcopy(A)
            g = copy.deepcopy(graph)
            for _ in range(parser.r - 1):
                data = np.dot(data, A)
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    if A[i, j] == 1:  # delete an edge
                        pij = len(list(nx.common_neighbors(graph, i, j))) / (
                                    len(list(list(g.neighbors(i)))) + len(list(g.neighbors(j))) + 2 - len(
                                list(nx.common_neighbors(graph, i, j)))) + np.dot(X[i, :], X[j, :]) / (
                                          np.linalg.norm(X[i, :]) * np.linalg.norm(X[j, :]))
                        pij = pij / 2
                        pij = pij + 1 / (1 + math.exp(
                            -(len(list(list(g.neighbors(i)))) + len(list(g.neighbors(j))) + 2) / parser.tau))
                        pij = math.exp(-pij / 2) / (1 + math.exp(-pij / 2))
                        p = np.random.rand()
                        if p <= pij:
                            g.remove_edge(i, j)
                            A[i, j] = 0
                            A[j, i] = 0
                    else:  # add an edge
                        pij = (data[i, j] / np.sum(data[i, :]) + np.dot(X[i, :], X[j, :]) / (
                                    np.linalg.norm(X[i, :]) * np.linalg.norm(X[j, :]))) / 2
                        pij = 1 / (2 + math.exp(-(1 - pij)))
                        p = np.random.rand()
                        if p <= pij:
                            g.add_edge(i, j)
                            A[i, j] = 1
                            A[j, i] = 1
            del data
            Y = X + np.random.normal(loc=0.0, scale=1, size=(X.shape[0], X.shape[1]))
            Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
            result.append([g, torch.FloatTensor(X)])
    return result