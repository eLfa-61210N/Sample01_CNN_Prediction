

import mnist

import numpy as np

from initializer import gen_kernal, gen_biases
from layer       import affine, conv
from pooling     import max_pool as pool
from transformer import one_hot
from activator   import softmax
from loss        import cross_entropy_error as cerror



img = mnist.train_images()
iml = mnist.train_labels()
iml = one_hot(iml[0])


# 使用例
print(img.shape)
layer_1_kernal = gen_kernal( 1, 2, 2)[0]
layer_1 = conv( img[0], layer_1_kernal)
layer_1 = pool( layer_1, [2,2])
print(layer_1)

layer_2_kernal = gen_kernal( 1, 2, 2)[0]
layer_2 = conv( layer_1, layer_2_kernal)
layer_2 = pool( layer_2, [2,2])
print(layer_2)

layer_3_kernal = gen_kernal( 128, layer_2.shape[0], layer_2.shape[1])
layer_3 = affine( layer_2, layer_3_kernal)
print(layer_3)

layer_4_kernal = gen_kernal( 10, 1, 128)
layer_4 = affine( layer_3, layer_4_kernal)
print(layer_4.shape)
print(layer_4)

layer_5 = softmax( layer_4)
print('sum chack', sum( layer_5[0]), ' ', layer_5)

learn_1 = cerror( layer_5, iml)
print(learn_1)


