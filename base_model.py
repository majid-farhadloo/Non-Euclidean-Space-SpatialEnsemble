#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

device = torch.device('cuda')


def knn(x, k):
    '''
    Gets the indices of the top K nearest neighbors of x
    '''
    inner = -2*torch.matmul(x.transpose(2, 1), x) # torch.Size([8, 4096, 4096])
    xx = torch.sum(x**2, dim=1, keepdim=True) # torch.Size([8, 1, 4096])
    pairwise_distance = -xx - inner - xx.transpose(2, 1) # torch.Size([8, 4096, 4096])
 
    idx = pairwise_distance.topk(k=k+1, dim=-1)[1][:,:,1:]   # (batch_size, num_points, k) : torch.Size([8, 4096, 10])

    return idx

class FCNN(nn.Module):
    def __init__(self, output_channels):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 128)
        self.fc7 = nn.Linear(128, 256)
        self.fc8 = nn.Linear(256, output_channels)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = torch.mean(x, dim=1)
        x = F.softmax(self.fc8(x), dim=1)
        return x, None, None


class PointNet(nn.Module):
    def __init__(self, args, output_channels):
        super(PointNet, self).__init__()
        self.args = args

        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, kernel_size=1, bias=False), 
                                   nn.BatchNorm1d(64), 
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=1, bias=False), 
                                   nn.BatchNorm1d(64), 
                                   nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=1, bias=False), 
                                   nn.BatchNorm1d(64), 
                                   nn.ReLU())

        self.conv4 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, bias=False), 
                                   nn.BatchNorm1d(128), 
                                   nn.ReLU())

        self.conv5 = nn.Sequential(nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False), 
                                   nn.BatchNorm1d(args.emb_dims), 
                                   nn.ReLU())

        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN(nn.Module):
    def __init__(self, args, output_channels):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k

        self.conv1 = nn.Sequential(nn.Conv2d(4, 64, kernel_size=1, bias=False), 
                                   nn.BatchNorm2d(64), 
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False), 
                                   nn.BatchNorm2d(64), 
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False), 
                                   nn.BatchNorm2d(128), 
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False), 
                                   nn.BatchNorm2d(256), 
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False), 
                                   nn.BatchNorm1d(args.emb_dims), 
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.get_knn_features(x, k=self.k, spatial_dims=2)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = self.get_knn_features(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = self.get_knn_features(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = self.get_knn_features(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

    def get_knn_features(self, x, k=20, spatial_dims=None, idx=None):
        '''
        Gets the features of the top K nearest neighbors of x
        '''
        batch_size, num_dims, num_points  = x.size()
        x = x.view(batch_size, -1, num_points)

        if spatial_dims is not None:
            x = x[:,:spatial_dims]
            num_dims = spatial_dims

        if idx is None:
            idx = knn(x, k=k)   # (batch_size, num_points, k)

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        target_features = x.view(batch_size*num_points, -1)[idx, :]
        target_features = target_features.view(batch_size, num_points, k, num_dims) # (batch_size, num_points, k, num_dims)
        
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1) # (batch_size, num_points, k, num_dims)
        features = torch.cat((target_features-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

        return features

