"""
    LSTM module
"""

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from utils import *


class lstm(nn.Module):
    def __init__(self, input_size=4, hidden_size=30, output_size=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 使用两层LSTMCell堆积来提高模型表达力
        self.layer1 = nn.LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size)
        self.layer2 = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        
        
    def forward(self, input_data, future=0):
        outputs = []
        
        # LSTM cell的三个输入：input(batch,input_size), h_0(batch,hidden_size), c_0(batch,hidden_size)
        # batch即为input_data中样本的数量，即为ntrain
        # 此处input_data的大小为：ntrain x SEQ_LENGTH X 4
        
        # init hidden states and cell state for layer1
        h_t = torch.zeros(input_data.size(0), self.hidden_size, dtype=torch.double)
        c_t = torch.zeros(input_data.size(0), self.hidden_size, dtype=torch.double)
        
        # init hidden states and cell state for layer2
        h_t2 = torch.zeros(input_data.size(0), self.hidden_size, dtype=torch.double)
        c_t2 = torch.zeros(input_data.size(0), self.hidden_size, dtype=torch.double)
        
        # input_data:[ntrain x SEQ_LENGTH X 4]
        # chunk将tensor按第二个维度分成 SEQ_LENGTH 块
        for i, input_t in enumerate(input_data.chunk(input_data.size(1), dim=1)):
            
            # reshape: [ntrain x 1 x 4] => [ntrain x 4] 
            input_t = input_t.squeeze(1)
            
            # 每个input_t是 ntrain x 4 的tensor， batch=ntrain，input_size=4
            h_t, c_t = self.layer1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.layer2(h_t, (h_t2, c_t2))
            
            # output的大小为 ntrainx1
            output = self.linear(h_t2)
            outputs += [output]
            
        outputs = torch.stack(outputs, 1).squeeze(2)
        
        return outputs
        
        
        
def train_LSTM():
    # 设置随机数种子
    np.random.seed(0)
    torch.manual_seed(0)

    # 构建网络
    Lstm = lstm().double()
    criterion = nn.MSELoss()

    # 优化器
    optimizer = optim.LBFGS(Lstm.parameters(), lr=0.1)

    # 准备数据
    xtrain, ytrain, xtest, ytest = LSTM_data()
    
    # 读取由GAN生成的数据
    gen_data = np.load('time_series.npy')
    gen_data = torch.from_numpy(gen_data).double()
    gen_xtrain = gen_data[:, :-1, :]
    gen_ytrain = gen_data[:, 1: , 2]
    
    # 开始训练    
    for i in range(100):
    
        # print('STEP: ', i)
    
        def closure():
            optimizer.zero_grad()
        
            out = Lstm(xtrain)
            loss = criterion(out, ytrain)
            # print('loss: %5.3f  ' %(loss.item()), end="")
            loss.backward()
        
            out_gen = Lstm(gen_xtrain)
            loss_gen = criterion(out_gen, gen_ytrain)
            # print('gen_loss: %5.3f' % (loss_gen.item()))
            loss_gen.backward()
            
            return loss + loss_gen
    
        optimizer.step(closure)
        
    return Lstm, criterion
        
        
def test_LSTM(Lstm, criterion):
    xtrain, ytrain, xtest, ytest = LSTM_data()
    
    with torch.no_grad():
        future = 0
        pred = Lstm(xtest, future=future)
        loss = criterion(ytest, pred)
        print('test loss:', loss.item())

    pred_data = pred.detach().numpy()
    test_data = ytest.detach().numpy()
    
    # 从每个batch的序列数据中提取出ground truth和预测值
    gd_truth = []
    pred_val = []

    # 每个batch的序列数据中的最后一个是新来的数据，将每个batch中新来的值提取出来
    for i in range(len(pred_data)):
        gd_truth.append(test_data[i][-1])
        pred_val.append(pred_data[i][-1])
    
    gd_truth = np.array(gd_truth)
    pred_val = np.array(pred_val)
    
    print("SCORE:", np.linalg.norm(gd_truth - pred_val))
