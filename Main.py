import matplotlib.pyplot as plt
import numpy as np
import random
from inout.plot import plot, distribution
from inout.fopen import dat_values
from model.random import selfRand, randNum, normNum
from model.shifts import peaks, shift
from model.trend import linear, expon
from analysis.probfun import sin
from analysis.stats import stationarity, statistics, correlation, harmonic_motion, fourier_transform,\
                           inverse_fourier_transform


def first_task(x, k, b, alp, bet):
    plot(linear(k, x, 0), desc='y = kx')
    plot(linear(-k, x, b), desc='y = -kx + b')
    plot(expon(alp, x), desc='y = e^ax')
    plot(expon(-bet, x), desc='y = e^-bx')


def second_task(N):
    plot(randNum(N), desc='Random numbers')
    plot(selfRand(N), desc='Self random numbers')
    plot(normNum(randNum(N), 1), desc='Normalize random numbers')
    plot(normNum(selfRand(N), 1), desc='Normalize self random numbers')


def third_task(x, rand):
    plot(shift(normNum(randNum(len(x)), 1), 1), desc='Shift for random')
    plot(shift(normNum(selfRand(len(x)), 1), 1), desc='Shift for self random')
    plot(peaks(0.001, 5, np.zeros(len(x))), desc='Peaks for zero function')
    stationarity(rand, 10, 0.05)


def fourth_task(n):
    print('Random stats')
    statistics(normNum(randNum(n), 1))
    print('Self random stats')
    statistics(normNum(selfRand(n), 1))
    distribution(normNum(randNum(n), 1), 'Random numbers histogram')
    distribution(normNum(selfRand(n), 1), 'Self random numbers histogram')


def fifth_task(n):
    tmp = np.zeros(100)
    for i in range(0, 100):
        tmp[i] = correlation(normNum(randNum(n), 1), normNum(randNum(n), 1), i)
    plot(tmp, desc='Корреляция')
    tmp = np.zeros(100)
    for i in range(0, 100):
        a = normNum(randNum(n), 1)
        tmp[i] = correlation(a, a, i)
    plot(tmp, desc='Автокорреляция')


def sixth_task(x, N):
    harm = harmonic_motion(x, 100, 37, 0.002)
    plot(harm, desc='Гармонический процесс')  # 3, 37, 137, 237, 337 [Гц]
    ft = fourier_transform(harm, N)
    #добавить в вектор ось Х
    plot([xx * 0.5 for xx in range(0, len(ft[1]) // 2)], ft[1][:len(ft[1]) // 2], desc='Преобразование Фурье')
    ift = inverse_fourier_transform(ft[0], N)
    plot(ift, desc='Обратное преобразование Фурье')


def seventh_task(dt, n):
    xarr = dat_values()
    xt = np.zeros(n)
    # f1 = sin(15, 3)
    # f2 = sin(100, 37)
    # f3 = sin(25, 137)
    # for (i, t) in zip(range(n), xarr):
    #     xt[i] = f1(t) + f2(t) + f3(t)
    # ft = fourier_transform(xt, n)
    harm = harmonic_motion(xarr, 100, 37, 0.002)
    plot(harm, desc='Гармонический процесс')
    ft = fourier_transform(xarr, n)
    plot([xx * 0.5 for xx in range(0, len(ft[1]) // 2)], ft[1][:len(ft[1]) // 2], desc='Преобразование Фурье')
    ift = inverse_fourier_transform(ft[0], N)
    plot([xx * 0.5 for xx in range(0, len(ift) // 2)], ift[:len(ift) // 2], desc='Обратное преобразование Фурье')


def eighteen_task(x, n):
    harmonic_autocorrelation = np.zeros(n)
    linear_autocorrelation = np.zeros(n)
    exponential_autocorrelation = np.zeros(n)
    harm = harmonic_motion(x, 100, 37, 0.002)
    for i in range(n):
        harmonic_autocorrelation[i] = correlation(harm, harm, i)
        linear_autocorrelation[i] = correlation(linear(1.3, x, 10000), linear(1.3, x, 10000), i)
        exponential_autocorrelation[i] = correlation(expon(0.0016, x), expon(0.0016, x), i)
    # # plot(tmp, desc='Автокорреляция гармонического процесса')
    # # distribution(harm, desc='Плотность вероятности гармонического процесса')
    # ft = fourier_transform(shift(selfRand(n), 0), n)
    # plot(ft[1], desc='Преобразование Фурье для случайного набора')
    # ift = inverse_fourier_transform(ft[0], N)
    # plot(ift, desc='Обратное преобразование Фурье для случайного набора')
    # # нет зависимостей от количества пиков
    plot(peaks(0.003, 2, np.zeros(len(harm))), desc='Peaks for zero function')
    # plot(harm, desc='Процесс')
    #бесокнечный набор синусов и косинусов даст 1.
    # distribution(linear(1.3, x, 10000), desc='Плотность вероятности линейной ф-ции')
    # distribution(expon(0.0016, x), desc='Плотность вероятности экспоненциальной ф-ции')
    # plot(linear_autocorrelation, desc='Автокорреляция линейной ф-ции')
    # plot(exponential_autocorrelation, desc='Автокорреляция экспоненциальной ф-ции')
    #

if __name__ == "__main__":
    fig = plt.figure(1)
    N = 1000
    k = random.uniform(2, N)
    b = random.uniform(1, N)
    alp = random.random()
    bet = random.random()
    x = np.arange(0, N)
    # first_task(x, 1.3, 1000, 0.0016, 6)
    # second_task(N)
    # rnd = randNum(N)
    # third_task(x, rnd)
    # fourth_task(N)
    # fifth_task(N)
    # sixth_task(x, N)
    # seventh_task(0.002, N)
    eighteen_task(x, N)
