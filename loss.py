

import numpy as np
from activator import softmax


# 二乗平均誤差
def sum_squared_error(answer, correct):
    y = answer
    t = correct
   
    E = (1/2) * np.sum((y - t) **2)
    return E

# 交差エントロピー誤差
def cross_entropy_error(answer, correct):
    
    y = answer
    t = correct
    D = 1e-6 # 無限大を防止
     
    E = -np.sum(t * np.log(y + D))
    return E


