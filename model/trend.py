import numpy as np


def linear(k, x, b):
    return [k * xx + b for xx in x]


def expon(a, x):
    return [np.exp(a*xx) for xx in x]
