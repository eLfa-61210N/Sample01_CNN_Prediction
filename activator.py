

import numpy as np


def step(x):
    if x > 0:
        return 1
    else:
        return 0


def relu(x):
    return x * (x > 0.0)


def softplus(x):
    return np.log( 1.0 + np.exp(x))


def tanh(x):
    return ( np.exp(x) - np.exp(-x)) / ( np.exp(x)) + np.exp(-x)


def mish(x):
    return x * tanh( softplus(x))


def identity(x):
    return x


def sigmoid(x):
    return 1.0 / ( 1.0 + np.exp(-x))


def softmax(u):
    exp_u = np.exp(u)
    exp_u_sum = np.sum(exp_u)
    y = exp_u / exp_u_sum
    return y



