import numpy as np
import random
import time


def stats(x, per):
    mean = []
    var = []
    for i in stationaryProcess(x, per):
        mean.append(np.mean(i))
        var.append(np.var(i))
    return {'Среднее арифметическое': mean,
            'Дисперсия случайной величины': var}


def stationaryProcess(x, per):
    n = len(x)
    res = []
    for i in range(per):
        res.append(x[i * int(n / per):((i + 1) * int(n / per))])
    return res


def normNum(x, s):
    return [(((xx - min(x))/(max(x) - min(x))) - 0.5) * 2 * s for xx in x]


def randNum(x, N):
    return [random.uniform(-1, N) for xx in x]


def selfRand(x):
    return [hash(time.clock() / (1 + xx)) for xx in x]

