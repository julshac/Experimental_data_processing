import numpy as np
import time
import math


def normNum(x, s):
    return (((x - min(x))/(max(x) - min(x))) - 0.5) * 2 * s


def randNum(n):
    return np.random.sample(n)


def selfRand(n):
    x = np.zeros(n)
    for i in range(n):
        x[i] = np.log(hash(time.clock()))*np.sin(i)
    return x

