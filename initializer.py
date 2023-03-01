

import numpy as np


# 新しくカーネルを生成する
def gen_kernal( quantity, height, width):
    """
        引数は(個数,縦,横)
    """

    kernal = np.random.rand( quantity, height, width) * 0.02
    return kernal


# 新しくバイアスを生成する
def gen_biases( size):
    biases = np.random.rand(1, size) * 0.01
    return biases


