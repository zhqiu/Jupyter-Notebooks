"""
    一些要用到的函数
"""

import numpy as np
import pandas as pd
import torch


# 将所有数据分割为训练集和测试集
# 与 LSTM -- predict stock price中的函数相同

def train_test_split(data, SEQ_LENGTH, test_prop=0.137):  # 0.11 for 1095, 0.137 for 730, 0.3 for 365
    
    ntrain = int(len(data) *(1-test_prop))  # len(data) = 197
    predictors = data.columns[:4]  # open, high, close, low
    data_pred = data[predictors]
    num_attr = data_pred.shape[1]  # 4
    
    result = np.empty((len(data) - SEQ_LENGTH, SEQ_LENGTH, num_attr))
    y = np.empty((len(data) - SEQ_LENGTH, SEQ_LENGTH))
    yopen = np.empty((len(data) - SEQ_LENGTH, SEQ_LENGTH))

    for index in range(len(data) - SEQ_LENGTH):
        result[index, :, :] = data_pred[index: index + SEQ_LENGTH]
        y[index, :] = data_pred[index+1: index + SEQ_LENGTH + 1].close
        yopen[index, :] = data_pred[index+1: index + SEQ_LENGTH + 1].open

    """
        xtrain的大小：ntrain x SEQ_LENGTH x 4
        ytrain的大小：ntrain x SEQ_LENGTH
        
        * xtrain的每个batch为长为SEQ_LENGTH的连续序列，一共有ntrain个batch，
          序列中每个单元都是一个四元组（open，high，close，low）
        * ytrain的每个batch为长为SEQ_LENGTH的连续序列，一共有ntrain个batch，
          序列中每个单元是xtrain中对应四元组所在日期的下一天的close price
        
        xtest 的大小：    ntest x SEQ_LENGTH x 4                
        ytest的大小：     ntest x SEQ_LENGTH      (close price)
        ytest_open的大小：ntest x SEQ_LENGTH      (open price)  
        
        * xtest的每个batch为长为SEQ_LENGTH的连续序列，一共有ntest个batch，
          序列中每个单元都是一个四元组（open，high，close，low）
          每一个序列仅包含一个新四元组，且在最后一个
        * ytest的每个batch为长为SEQ_LENGTH的连续序列，一共有ntest个batch，
          序列中每个单元是xtest中对应四元组所在日期的下一天的close price
        
        类型：numpy.ndarray
    """
    xtrain = result[:ntrain, :, :]
    ytrain = y[:ntrain]
    
    xtest = result[ntrain:, :, :]
    ytest = y[ntrain:]
    ytest_open = yopen[ntrain:]
    
    return xtrain, xtest, ytrain, ytest, ytest_open
    
    
    
# 准备GAN所需的数据
# 从csv文件中读取数据，将数据划分为训练集和测试集
# 再从训练集中随机抽取m个长度为n的串返回
def prepare_data():
    data = pd.read_csv('stock_data_730.csv')

    data.set_index(["date"], inplace=True)
    data_sorted = data.sort_index()

    # 只需要xtrain    

    xtrain, xtest, ytrain, ytest, ytest_open = train_test_split(data_sorted, SEQ_LENGTH=26)

    xtrain_mean = np.mean(xtrain)
    xtrain_max = np.max(xtrain)
    xtrain_min = np.min(xtrain)
    
    return xtrain, xtrain_mean, xtrain_max, xtrain_min
    
    
   
# 提供LSTM所需的数据
def LSTM_data():
    data = pd.read_csv('stock_data_730.csv')

    data.set_index(["date"], inplace=True)
    data_sorted = data.sort_index()
    
    xtrain, xtest, ytrain, ytest, ytest_open = train_test_split(data_sorted, SEQ_LENGTH=25)
    
    # 转为tensor
    xtrain = torch.from_numpy(xtrain)
    ytrain = torch.from_numpy(ytrain)

    xtest = torch.from_numpy(xtest)
    ytest = torch.from_numpy(ytest)
    
    return xtrain, ytrain, xtest, ytest