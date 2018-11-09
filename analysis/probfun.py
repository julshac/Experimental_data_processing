import numpy as np
import math


def _deviation(x):
    return np.sqrt(_variance(x))


def _subtraction_for_correlation(x):
    t = 0
    for i in range(len(x)):
        t += (x[i] - x.mean()) ** 2
    return t


def _error(x):
    return np.sqrt(_rms(x))


def _average(x):
    return _sum(x, 1, 0) / len(x)


def _variance(x):
    return (x**2).sum() / len(x)


#   root mean square
def _rms(x):
    return _sum(x, 2, 0) / len(x)


def _skewness(x):
    return _sum(x, 3, _average(x)) / len(x)


def _kurtosis(x):
    return _sum(x, 4, _average(x)) / len(x)


def _gamma(expVal, dev, power, tmp):
    return (expVal / (dev ** power)) - tmp


def _sum(x, power, temp):
    sm = 0
    for i in x:
        sm += (i - temp) ** power
    return sm


def sin(a, f):
    return lambda t: a * math.sin(2 * math.pi * f * t)
