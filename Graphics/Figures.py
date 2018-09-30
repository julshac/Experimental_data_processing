import numpy as np


def fig_1(x, k):
    return [k*xx for xx in x]


def fig_2(k, x, b):
    return [-k * xx + b for xx in x]


def fig_3(alp, x):
    return [np.exp(alp*xx) for xx in x]


def fig_4(bet, x):
    return [np.exp(-bet*xx) for xx in x]


def temp(x, N, bet):
    k = np.arange(2, N, 0.01)
    for kk in k:
        if k * x - np.exp(-bet * x) < 0.00001:
            return kk