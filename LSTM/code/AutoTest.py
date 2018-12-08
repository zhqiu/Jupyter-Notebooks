"""
    自动测试
"""

import numpy as np
from GAN import *
from LSTM import *


"""
    构造，训练GAN
    将生成的数据保存起来
"""

# 数据的batch_size
gen_batch_size = 100
# 输入噪声的长度
input_size = 20 


print("start trainning GAN...")
G, xtrain_mean, xtrain_max, xtrain_min = GAN()
print("finished.")

print("start generating data...")
input_data = get_noise_data(gen_batch_size, input_size)
time_series = G.forward(input_data)
time_series = time_series.permute(0,2,1).detach().numpy()
time_series = time_series * (xtrain_max - xtrain_min) + xtrain_mean
np.save('./time_series.npy', time_series)
print("finished.")



"""
    利用原始数据及生成数据训练LSTM
    并在测试集上进行测试得到得分
"""
print("start trainning LSTM...")
lstm, criterion = train_LSTM()
print("finished.")
test_LSTM(lstm, criterion)
