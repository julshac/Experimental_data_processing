import numpy as np
import random
import time


def stats(x):
    mean = x.mean()
    std = x.std()
    var = x.var()
    return {'Среднее арифметическое': mean,
            'Среднеквадратическое отклонение': std,
            'Дисперсия случайной величины': var}


def normNum(x, s):
    return [(((xx - min(x))/(max(x) - min(x))) - 0.5) * 2 * s for xx in x]


def stationaryProcess(x, per):
    n = len(x)
    res = []
    for i in range(per):
        res.append(x[i * (n / per):((i + 1) * (n / per))])
    return res


def randNum(x, N):
    return [random.uniform(-1, N) for xx in x]


def selfRand(x):
    return [hash(time.clock() / (1 + xx)) for xx in x]

