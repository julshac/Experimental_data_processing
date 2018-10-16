import numpy as np
from analysis.probfun import _deviation, _skewness, _kurtosis, _rms, _variance, _average, _error, _gamma, _sum


def spacing(x, per):
    n = len(x)
    res = []
    for i in range(per):
        res.append(x[i * int(n / per):((i + 1) * int(n / per))])
    return res


def stationarity(x, per, perc=0.01):
    means = np.zeros(per)
    var = np.zeros(per)
    length_of_int = len(x)//per
    for i in range(per):
        means[i] = x[i * length_of_int:(i + 1) * length_of_int].mean()
        var[i] = x[i * length_of_int:(i + 1) * length_of_int].std()
    stat = True
    for m in means:
        if abs(m - means.mean()) > abs(means.mean()) * perc:
           print("Ошибочное среднее значение  {}, при {} > {}".format(m, abs(m - means.mean()),
                                                                        abs(means.mean()) * perc))
           stat = False
    for v in var:
        if abs(v - var.mean()) > abs(var.mean()) * perc:
           print("Ошибочное значение дисперсии {}, при {} > {}".format(v, abs(v - var.mean()),
                                                                       abs(var.mean()) * perc))
           stat = False
    print("Стационарен ли процесс? {}".format(stat))
    print("Стредние значения:")
    print(means)
    print("Дисперсии:")
    print(var)
    # print('СКО средних значений по 10 замерам: {0}'.format(np.means(means)))
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


#ллинейный коэффициент корреляции Rxx
def rxx(x, y, shift):
    tmp = 0
    for i in range(len(x) - shift):
        tmp += ((x[i] - x.mean())*(y[i + shift] - y.mean()))
    return tmp


def covariance(x, y, shift):
    return rxx(x, y, shift) / len(x)


def correlation(x, y, shift):
    return rxx(x, y, shift) / _deviation(x)
#    return np.correlate(rnd, srnd, mode='full')


#Гармонический процесс
def harmonic_motion(x, a, f, t):
    return a * np.sin(2 * np.pi * f * x * t)


#Преобразование Фурье
def fourier_transform(x, N):
    cs = []
    cn = []
    # x = list(map(float, x))
    for n in range(0, N):
        re = 0
        im = 0
        for k in range(N):
            re += x[k] * np.cos((2 * np.pi * n * k) / N)
            im += x[k] * np.sin((2 * np.pi * n * k) / N)
        re /= N
        im /= N
        cs.append(re + im)
        cn.append(np.sqrt(re ** 2 + im ** 2))
    return cs, cn


#Обратное преобразование Фурье
def inverse_fourier_transform(cs, N):
    xk = []
    for n in range(0, N):
        re = 0
        im = 0
        for k in range(N):
            re += cs[k] * np.cos((2 * np.pi * n * k) / N)
            im += cs[k] * np.sin((2 * np.pi * n * k) / N)
        xk.append(re + im)
    return xk

