import matplotlib.pyplot as plt
import numpy as np
import random
from inout.plot import vector, distribution
from model.random import selfRand, randNum, normNum
from model.shifts import peaks, shift
from model.trend import linear, expon
from analysis.stats import stationarity, statistics, correlation, harmonic_motion, fourier_transform, inverse_fourier_transform


def first_task(x, k, b, alp, bet):
    vector(linear(k, x, 0), 'y = kx')
    vector(linear(-k, x, b), 'y = -kx + b')
    vector(expon(alp, x), 'y = e^ax')
    vector(expon(-bet, x), 'y = e^-bx')


def second_task(N):
    vector(randNum(N), 'Random numbers')
    vector(selfRand(N), 'Self random numbers')
    vector(normNum(randNum(N), 1), 'Normalize random numbers')
    vector(normNum(selfRand(N), 1), 'Normalize self random numbers')


def third_task(x, rand):
    vector(shift(normNum(randNum(len(x)), 1), 1), 'Shift for random')
    vector(shift(normNum(selfRand(len(x)), 1), 1), 'Shift for self random')
    vector(peaks(0.001, 5, np.zeros(len(x))), 'Peaks for zero function')
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
    vector(tmp, 'Корреляция')
    tmp = np.zeros(100)
    for i in range(0, 100):
        a = normNum(randNum(n), 1)
        tmp[i] = correlation(a, a, i)
    vector(tmp, 'Автокорреляция')
    # vector(autocorrelation(normNum(selfRand(x), 1)), 'Автокорреляция')


def sixth_task(x, N):
    harm = harmonic_motion(x, 100, 37, 0.002)
    vector(harm, 'Гармонический процесс') #3, 37, 137, 237, 337 [Гц]
    ft = fourier_transform(harm, N)
    #добавить в вектор ось Х
    vector([xx * 0.5 for xx in range(0, len(ft[1])//2)], ft[1][:len(ft[1])//2], 'Преобразование Фурье')
    ift = inverse_fourier_transform(ft[0], N)
    vector(ift, 'Обратное преобразование Фурье')


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
    sixth_task(x, N)

