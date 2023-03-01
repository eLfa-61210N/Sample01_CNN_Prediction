
import logging

import numpy  as np
import neuron as nrn
from transformer import tensor_transform




def conv( input_tensor, kernal, strides=[1,1], activtype='relu'):
    """注
        複数枚カーネルの畳み込みはここでは処理せず
        loop処理を使う
        input_tensor、kernalは2次元
    """
    """処理
        受け取ったテンソルを変形し
        変形したテンソルとカーネルをニューロンに入力する
        ニューロンの出力を配列後整形し出力する
    """
    output_tensor = []
    size_kernal_y = kernal.shape[0]
    size_kernal_x = kernal.shape[1]
    strides_y = strides[0]
    strides_x = strides[1]
    size_feature_map_y = ( ( input_tensor.shape[0] -size_kernal_y) // strides_y) +1
    sfmy = size_feature_map_y
    size_feature_map_x = ( ( input_tensor.shape[1] -size_kernal_x) // strides_x) +1
    sfmx = size_feature_map_x

    transed_tensor = tensor_transform( input_tensor, kernal.shape, strides)
    transed_kernal = tensor_transform( kernal, kernal.shape, [1,1])
    
    cache = []
    for tensor in transed_tensor:
        n = nrn.neuron( tensor, transed_kernal[0], 0, activtype)
        cache.append(n)
    output_tensor = np.array( [cache])
    output_tensor = output_tensor.reshape( sfmy, sfmx)

    feature_map = output_tensor

    return feature_map



def affine( input_tensor, kernal, activtype='relu'):
    """
        input_tensorは2次元、kernalは3次元
    """
    """
        全結合層では入力されたテンソルとカーネルをニューロンに入力し
        ニューロンの出力を配列し
        カーネルの個数個のノードを持つベクトルを出力する
    """

    if input_tensor.shape != kernal[0].shape:
        logging.warning( 'isn\'t affine , call function convolution')
        print(f'ts shape: {input_tensor.shape}    kernal shape: {kernal[0].shape}')

    output_vector = np.array([])

    cache = []
    input_vector = input_tensor.reshape( 1 ,input_tensor.shape[0] *input_tensor.shape[1])[0] # 最後の[0]reshapeを一段外す
    for knl_num in range(kernal.shape[0]):
        knl = kernal[ knl_num].reshape( 1, kernal.shape[1] *kernal.shape[2])[0] # 最後の[0]はreshapeを一段外す
        cache.append( nrn.neuron( input_vector, knl, 0, activtype))
    output_vector = np.append( output_vector, cache)
    output_vector = output_vector.reshape(1,output_vector.shape[0])

    return output_vector

#print(f'tensor out  : {conv(img[0], tens1[1], [1,1], "relu")}')





