import numpy as np
import math
from analysis.probfun import _deviation, _skewness, _kurtosis, _rms, _variance, _average, _error, _gamma


def spacing(x, per):
    n = len(x)
    res = []
    for i in range(per):
        res.append(x[i * int(n / per):((i + 1) * int(n / per))])
    return res


def stationarity(x, per, perc=0.5):
    mean = [_average(i) for i in spacing(x, per)]
    var = [_variance(i) for i in spacing(x, per)]
    stat = True
    for m in mean:
        if abs(m - _average(mean)) > _average(mean) * perc:
           print("Ошибочное среднее значение  {}, при {} > {}".format(m, abs(m - _average(mean)),
                                                                        _average(mean) * perc))
           stat = False
    for v in var:
        if abs(v - _average(var)) > _average(mean) * perc:
           print("Ошибочное значение дисперсии {}, при {} > {}".format(v, abs(v - _average(var)),
                                                             _average(var) * perc))
           stat = False
    print("Стационарен ли процесс? {}".format(stat))
    print("Стредние значения:")
    print(mean)
    print("Дисперсии:")
    print(var)
    # print('СКО средних значений по 10 замерам: {0}'.format(np.mean(mean)))
    # print('СКО дисперсий по 10 замерам: {0}'.format(np.var(var)))


def statistics(rnd):
    print('Среднее арифметическое: {0}'.format(_average(rnd)))
    print('Дисперсия случайной величины: {0}'.format(_variance(rnd)))
    print('Средний квадрат: {0}'.format(_rms(rnd)))
    print('Среднее отклонение: {0}'.format(_deviation(rnd)))
    print('Средне квадратичная ошибка: {0}'.format(_error(rnd)))
    print('Эксцесс: {0}'.format(_kurtosis(rnd)))
    print('Коэффициент ассиметрии: {0}'.format(_skewness(rnd)))
    print('Гамма 1: {0}'.format(_gamma(_skewness(rnd), _deviation(rnd), 3, 0)))
    print('Гамма 2: {0}'.format(_gamma(_skewness(rnd), _deviation(rnd), 4, 3)))


def per_statistics(rnd, per):
    print('Среднее арифметическое: {0}'.format(np.std([_average(i) for i in spacing(rnd, per)])))
    print('Дисперсия случайной величины: {0}'.format(np.std([_variance(i) for i in spacing(rnd, per)])))
    print('Средний квадрат: {0}'.format(np.std([_rms(i) for i in spacing(rnd, per)])))
    print('Среднее отклонение: {0}'.format(np.std([_deviation(i) for i in spacing(rnd, per)])))
    print('Средне квадратичная ошибка: {0}'.format(np.std([_error(i) for i in spacing(rnd, per)])))
    print('Эксцесс: {0}'.format(np.std([_kurtosis(i) for i in spacing(rnd, per)])))
    print('Коэффициент ассиметрии: {0}'.format(np.std([_skewness(i) for i in spacing(rnd, per)])))
    print('Гамма 1: {0}'.format(np.std([_gamma(_skewness(i), _deviation(i), 3, 0) for i in spacing(rnd, per)])))
    print('Гамма 2: {0}'.format(np.std([_gamma(_skewness(i), _deviation(i), 4, 3) for i in spacing(rnd, per)])))


def covariance(x):
    return np.cov(x)


def correlation(rnd, srnd):
    return np.correlate(rnd, srnd, mode='full')


def autocorrelation(rnd):
    auto = correlation(rnd, rnd)
    return auto[auto.size//2:]
