# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:58:01 2019

@author: WT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class gcn(nn.Module):
    def __init__(self, X_size, A_hat, hidden_1, hidden_2, no_classes, bias=True):  # X_size = num features
        super(gcn, self).__init__()
        self.A_hat = torch.tensor(A_hat, requires_grad=False).float().cuda()
        self.weight = nn.parameter.Parameter(torch.FloatTensor(X_size, hidden_1))
        var = 2. / (self.weight.size(1) + self.weight.size(0))
        self.weight.data.normal_(0, var)
        self.weight2 = nn.parameter.Parameter(torch.FloatTensor(hidden_1, hidden_2))
        var2 = 2. / (self.weight2.size(1) + self.weight2.size(0))
        self.weight2.data.normal_(0, var2)
        if bias:
            self.bias = nn.parameter.Parameter(torch.FloatTensor(hidden_1))
            self.bias.data.normal_(0, var)
            self.bias2 = nn.parameter.Parameter(torch.FloatTensor(hidden_2))
            self.bias2.data.normal_(0, var2)
        else:
            self.register_parameter("bias", None)
        self.fc1 = nn.Linear(hidden_2, no_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X):  ### 2-layer GCN architecture
        X = torch.mm(X, self.weight)
        # print(X.shape)
        if self.bias is not None:
            X = (X + self.bias)
        X = F.relu(torch.mm(self.A_hat, X))
        # print(X.shape)
        X = torch.mm(X, self.weight2)
        # print(X.shape)
        if self.bias2 is not None:
            X = (X + self.bias2)
        X = F.relu(torch.mm(self.A_hat, X))
        # print(X.shape)
        return self.softmax(self.fc1(X))