import matplotlib.pyplot as plt
import numpy as np
import random
from inout.plot import vector, distribution
from model.random import selfRand, randNum, normNum
from model.shifts import peaks, shift
from model.trend import linear, expon
from analysis.stats import stationarity, statistics, covariance, correlation, autocorrelation


def first_task(x, k, b, alp, bet):
    vector(linear(k, x, 0), 'y = kx')
    vector(linear(-k, x, b), 'y = -kx + b')
    vector(expon(alp, x), 'y = e^ax')
    vector(expon(-bet, x), 'y = e^-bx')


def second_task(x):
    vector(randNum(x), 'Random numbers')
    vector(selfRand(x), 'Self random numbers')
    vector(normNum(randNum(x), 1), 'Normalize random numbers')
    vector(normNum(selfRand(x), 1), 'Normalize self random numbers')


def third_task(x, rand, n):
    vector(shift(normNum(randNum(x), 1), 1), 'Shift for random')
    vector(shift(normNum(selfRand(x), 1), 1), 'Shift for self random')
    vector([peaks(0.001, 1) for xx in range(0, n)], 'Peaks for zero function')
    stationarity(rand, 10)


def fourth_task():
    print('Random stats')
    statistics(normNum(randNum(x), 1))
    print('Self random stats')
    statistics(normNum(selfRand(x), 1))
    distribution(normNum(randNum(x), 1), 'Random numbers histogram')
    distribution(normNum(selfRand(x), 1), 'Self random numbers histogram')


def fifth_task(x):
    print(covariance(normNum(randNum(x), 1)))
    # print(correlation(normNum(randNum(x), 1)))
    # print(autocorrelation(normNum(randNum(x), 1)))
    # vector(covariance(normNum(randNum(x), 1)), 'Ковариация')
    vector(correlation(normNum(randNum(x), 1), normNum(selfRand(x), 1)), 'Корреляция')
    vector(autocorrelation(normNum(selfRand(x), 1)), 'Автокорреляция')


if __name__ == "__main__":
    fig = plt.figure(1)
    N = 100
    k = random.uniform(2, N)
    b = random.uniform(1, N)
    alp = random.random()
    bet = random.random()
    x = np.arange(0, N)
    # first_task(x, k, b, alp, bet)
    # second_task(x)
    # third_task(x, normNum(selfRand(x), 1), N)
    # fourth_task()
    fifth_task(x)

