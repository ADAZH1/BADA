import matplotlib.pyplot as plt
import numpy as np

data = np.arange(1, 25)
plt.plot(data, data**2, color = 'r', marker = 'o', linestyle = '-.', alpha=0.5)
plt.savefig(r"/home/zhangheng/SENTRYTQS/Picture/test.jpg")
