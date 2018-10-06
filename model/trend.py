import numpy as np


def linear(k, x, b):
    return [k * xx + b for xx in x]


def expon(a, x):
    return [np.exp(a*xx) for xx in x]


def temp(x, N, bet):
    k = np.arange(2, N, 0.01)
    for kk in k:
        if k * x - np.exp(-bet * x) < 0.00001:
            return kk