import numpy as np
from torchvision import models

empty_array = np.array([])

# 向空数组中添加数据
data= np.array([ [i*2,i*3] for i in range(10)]).flatten()
data = np.append(data, [1,2,3])

print(data)