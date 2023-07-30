import numpy
import numpy as np
import torch
from numpy import *
# m1 = np.array(np.arange(0,7, dtype = float))
# m1 = m1.reshape(1,7)
# m1 = torch.from_numpy(m1)
# print(m1)
#
# m2 = np.array(np.arange(8,15, dtype = float))
# m2 = m2.reshape(1,7)
# m2 = torch.from_numpy(m2)
# print(m2)
#
# matrix =torch.dist(m1, m2) # 只能两者之间比较？
# print(matrix)

m1 = torch.array(np.arange(0, 24, dtype=float))
m1 = m1.reshape(6, 4)  # 计算三行之间的dist

num = 0
dist = torch.zeros((m1.shape[0], m1.shape[0]))

print(m1.shape[0])
for i in range(0, m1.shape[0]):
    for j in range(0, m1.shape[0]):
        dist[i][j] = torch.dist(m1[i], m1[j])

print(dist)



tempdist = torch.zeros(dist.shape[0])
for i in range(0, dist.shape[0]):
    # 对于每一行计算当前的最近的K个 现在认为是4（去除本身的其实就是3个）欧几里得距离 并求解出和
    dist[i] = sorted(dist[i])
    for j in range(0, 50):# 取到前面4个 但还要去除本身的 所以还是前三个最小的 所以这边就是参数 选择最近的几个
        tempdist[i] += dist[i][j]
print(dist)
print(tempdist)
index = argsort(-tempdist)
print(index)