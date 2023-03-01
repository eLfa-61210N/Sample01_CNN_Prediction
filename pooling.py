
import numpy as np
from transformer import tensor_transform

# 最大値プーリング
def max_pool( input_tensor, filter_size):
    """
        input_tensor  2次元
        filter_size   2次元
        output_tensor 2次元
    """

    output_tensor = np.array([])
    ts = input_tensor
    fy = filter_size[0]
    fx = filter_size[1]
    strides = [fy, fx]
    

    transed_tensor = tensor_transform( input_tensor, filter_size, strides )

    cache = []
    for tensor in transed_tensor:
        cache.append( max(tensor))
    output_tensor = np.array( [cache])

    output_tensor = output_tensor.reshape( ts.shape[0]//strides[0], ts.shape[1]//strides[1])

    return output_tensor




