

import numpy as np


# 'd'で始まる関数は微分したもの

def identity(x):
    return x


def d_identity(x):
    return 1


def step(x):
    if x > 0:
        return 1
    else:
        return 0


def d_step(x):
    return 0


def relu(x):
    return x * (x > 0.0)


def d_relu(x):
    return step(x)


def relu6(x):
    if x <= 0:
        return 0
    elif x >= 6:
        return 6
    else:
        return x


def d_relu6(x):
    if ( x <=  0) | ( 6 <= x):
        return 0
    else:
        return 1


def sigmoid(x):
    return 1.0 / ( 1.0 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * ( 1.0 - sigmoid(x) )


def hardsigmoid(x):
    if x > 2.5:
        return 1
    elif x < -2.5:
        return 0
    else:
        return (0.2 * x) + 0.5


def d_hardsigmoid(x):
    if  -2.5 <= x <= 2.5:
        return 0.2
    else:
        return 0


def log_sigmoid(x):
    return np.log( 1.0 / ( 1.0 + np.exp(-x)))


def d_log_sigmoid(x):
    return 1 / ( 1 + np.exp(x))


def softplus(x):
    return np.log( 1.0 + np.exp(x))


def d_softplus(x):
    return 1 / ( 1 + np.exp(-x))


def tanh(x):
    return ( np.exp(x) - np.exp(-x)) / ( np.exp(x)) + np.exp(-x)


def d_tanh(x):
    return 4 / (( np.exp(x) + np.exp(-x)) **2)


def tanh_shrink(x):
    return x - tanh(x)


def d_tanh_shrink(x):
    return tanh(x) **2


def hardtanh(x):
    if x > 1:
        return 1
    elif x < -1:
        return -1
    else:
        return x


def d_hardtanh(x):
    if ( x < -1) | ( 1 <= x):
        return 0
    else:
        return 1


def tanhexp(x):
    return x * tanh( np.exp(x))


def d_tanhexp(x):
    return tanh( np.exp(x)) - ( x * np.exp(x) * ( ( tanh( np.exp(x)) **2) -1))


def mish(x):
    return x * tanh( softplus(x))


def d_mish(x):
    z = ( 4 * ( x +1)) + ( 4 * np.exp( x *2)) + np.exp( x *3) + ( ( 4 * x) +6) * np.exp(x)
    d = 2 * np.exp(x) + np.exp( x *2) +2
    return ( np.exp(x) * z) / ( d **2)


def softmax(u):
    exp_u = np.exp(u)
    exp_u_sum = np.sum(exp_u)
    y = exp_u / exp_u_sum
    return y


def d_softmax(u):
    y = softmax(u)
    jcp = -y[ :, :, None] * y[ :, None, :]
    i_y, i_x = np.diag_indices_from(jcb[0])
    jcb[ :, i_y, i_x] = y * ( 1.0 -y)
    return jcb






