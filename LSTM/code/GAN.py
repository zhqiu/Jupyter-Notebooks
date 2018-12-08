"""
    GAN module
"""

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal

from utils import *


# 产生用于输入Generator的噪声, 为正态分布
# 其大小为 m x n
def get_noise_data(m, n):
    dist = Normal(0, 1)
    return dist.sample((m, n)).requires_grad_()
    
   
   
class Generator(nn.Module):
    def __init__(self, input_size=20, num_features=4, batch_size=10, seq_len=26):
        super().__init__()
        self.input_size = input_size   # nums of input rand numbers
        self.num_features = num_features
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.output_size = self.num_features * self.seq_len
        
        # 使用MLP
        self.fc1 = nn.Linear(self.input_size, self.output_size*10)
        self.fc2 = nn.Linear(self.output_size*10, self.output_size*5)
        self.fc3 = nn.Linear(self.output_size*5, self.output_size)
        
    def forward(self, input_data):
        output = self.fc1(input_data)
        output = self.fc2(output)
        output = torch.sigmoid(self.fc3(output))
        
        # output size: [batch_size, channels=num_features, width=seq_len]
        output = output.view(output.size(0), self.num_features, self.seq_len)
        
        return output
        
        
        
class Discriminator(nn.Module):
    def __init__(self, batch_size=10, seq_len=26, num_features=4):
        super().__init__()
        self.input_size = seq_len * num_features
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        # 使用 CNN
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(640, 256),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, input_data):
        # input_data的大小：[batch_size, channels=4, width=26]
        output = self.conv1(input_data)
        output = self.conv2(output)
        output = self.conv3(output)
        output = output.view(output.size(0), -1) # [batch_size, 640]
        output = self.fc1(output)
        output = self.fc2(output)
        output = torch.sigmoid(self.fc3(output))
        
        return output



# 获取下标从start_idx开始的连续batch_size个序列
def get_real_samples(idx, batch_size, data):
    data = data[idx:idx+batch_size, :]
    min_val = np.min(data)
    max_val = np.max(data)
    mean_val = np.mean(data)
    data = (data-mean_val)/(max_val-min_val)
    
    data = torch.from_numpy(data).float()
    data = data.view(batch_size, 4, -1)
    
    return data.requires_grad_()
    
    
    
# 构造GAN，并训练
# 返回训练好的Generator
def GAN():
    d_learning_rate = 0.01
    g_learning_rate = 0.01

    input_size = 20
    batch_size = 5
    seq_len = 26

    G = Generator(input_size=input_size, batch_size=batch_size, seq_len=seq_len)
    D = Discriminator(batch_size=batch_size, seq_len=seq_len)

    d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate) 
    g_optimizer = optim.SGD(G.parameters(), lr=g_learning_rate)
    
    xtrain, xtrain_mean, xtrain_max, xtrain_min = prepare_data()

    num_epochs = 100

    for epoch in range(num_epochs):
        G.train()
        D.train()
    
        sum_real_error = 0
        sum_gen_loss = 0
    
        start_time = time.time()
    
        for idx in range(0, len(xtrain)-batch_size+1, batch_size):
        
            """
                训练Discriminator
            """
            D.zero_grad()
            # 用真实样本训练D 
            real_data = get_real_samples(idx, batch_size, xtrain)
            real_decision = D(real_data)
            real_error = -torch.sum(torch.log(real_decision))/batch_size
            sum_real_error += real_error
            real_error.backward()
    
            # 用生成样本训练D
            input_data = get_noise_data(batch_size, input_size)
            fake_data = G.forward(input_data) 
            fake_decision = D(fake_data)
            fake_error = 1 - torch.sum(torch.log(fake_decision))/batch_size
            fake_error.backward()
    
            d_optimizer.step()
    
    
            """
                训练Generator
            """
            G.zero_grad()
            # 训练G
            input_data = get_noise_data(batch_size, input_size)
            fake_data = G.forward(input_data)
            fake_decision = D(fake_data)
            gen_loss = -torch.sum(torch.log(fake_decision))/batch_size
            sum_gen_loss += gen_loss
            gen_loss.backward()

            g_optimizer.step()
    
        # print("Epoch %5d:  D Loss  %5.3f  " % (epoch, sum_real_error), end="")
        # print("G Loss  %5.3f  " % (sum_gen_loss), end="")
        # print("duration: %5f" %(time.time()-start_time))
        
    return G, xtrain_mean, xtrain_max, xtrain_min