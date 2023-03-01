

import logging

import numpy as np
import activator as activ


# 人工ニューロン
def neuron( neuron_inputs, weights, bias, activtype='relu'):
    """
        重み付き線形和に活性化関数を適用したものを
        活性化関数別に出力
    """

    # テンソルとカーネルのサイズ違いを警告する
    if len( neuron_inputs) != len( weights):
        logging.error( 'neuron input data length error')
        exit()


    if activtype == 'relu':
        return activ.relu( w_l_summation( neuron_inputs, weights, bias))
    elif activtype == 'step':
        return activ.step( w_l_summation( neuron_inputs, weights, bias))
    elif activtype == 'mish':
        return activ.mish( w_l_summation( neuron_inputs, weights, bias))
    elif activtype == 'identity':
        return activ.identity( w_l_summation( neuron_inputs, weights, bias))
    elif activtype == 'sigmoid':
        return activ.sigmoid( w_l_summation( neuron_inputs, weights, bias))
    elif activtype == 'softmax':
        logging.warning('softmax関数は単独で最後に適用してください。')
        return activ.softmax( w_l_summation( neuron_inputs, weights, bias))
    else:
        logging.error( 'unknown activator')
        exit()


# weighted linear summation(重み付き線形和)
def w_l_summation( input_x, weights, bias):

    out_y = 0

    for x_i, w_i in zip( input_x, weights):
        out_y += x_i * w_i
    
    out_y = out_y +bias

    return out_y



