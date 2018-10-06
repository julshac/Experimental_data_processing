import numpy as np


def station(x, per):
    n = len(x)
    mean = []
    var = []
    std2 = []
    delt = []
    mu3 = []
    mu4 = []
    gamma1 = []
    gamma2 = []
    mu3.append(_mu3(stationaryProcess(x, per), n, mean))
    mu4.append(_mu4(stationaryProcess(x, per), n, mean))
    for i in stationaryProcess(x, per):
        mean.append(np.mean(i))
        var.append(np.var(i))
        std2.append(std2(i, n))
        delt.append(np.sqrt(var[i]))
        gamma1 = mu3[i] / delt[i] ** 3
        gamma2 = mu4[i] / delt[i] ** 4 - 3
    return {'Среднее арифметическое': mean,
            'Дисперсия случайной величины': var,
            'Средний квадрат': std2,
            'Эксцесс': mu4,
            'Центральный процесс 3его порядка': mu3,
            'Гамма 1': gamma1,
            'Гамма 2': gamma2}


def statistics(x):
    n = len(x)
    mean = np.mean(x)
    var = np.var(x)
    std2 = _std2(x, n)
    delt = np.sqrt(var)
    mu3 = (_mu3(x, n, mean))
    mu4 = (_mu4(x, n, mean))
    gamma1 = mu3 / delt ** 3
    gamma2 = mu4 / delt ** 4 - 3
    return {'Среднее арифметическое': mean,
            'Дисперсия случайной величины': var,
            'Средний квадрат': std2,
            'Эксцесс': mu4,
            'Центральный процесс 3его порядка': mu3,
            'Гамма 1': gamma1,
            'Гамма 2': gamma2}


def _std2(x, n):
    return 1 / n * _sum(x, 2, 0)


def _sum(x, power, temp):
    sm = 0
    for i in x:
        sm += i ** power - temp
    return sm


def _mu3(x, n, mean):
    return 1 / n * _sum(x, 3, mean)


def _mu4(x, n, mean):
    return 1 / n * _sum(x, 4, mean)


def stationaryProcess(x, per):
    n = len(x)
    res = []
    for i in range(per):
        res.append(x[i * int(n / per):((i + 1) * int(n / per))])
    return res
