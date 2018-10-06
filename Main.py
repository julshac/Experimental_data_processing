import matplotlib.pyplot as plt
import numpy as np
import random
from io.plot import vector, histogram
from model.random import selfRand, randNum, normNum
from model.trend import linear, expon
from analysis.stats import station, statistics


def first_task(x, k, b, alp, bet):
    vector(linear(k, x, 0), 'y = kx')
    vector(linear(-k, x, b), 'y = -kx + b')
    vector(expon(alp, x), 'y = e^ax')
    vector(expon(-bet, x), 'y = e^-bx')


def second_task(x, n):
    vector(randNum(x, n), 'Random numbers')
    vector(selfRand(x), 'Self random numbers')
    vector(normNum(randNum(x, n), 1), 'Normalize random numbers')
    vector(normNum(selfRand(x), 1), 'Normalize self random numbers')


def third_task(x, n):
    histogram(normNum(randNum(x), 1), 'Random numbers histogram', 100)
    histogram(normNum(selfRand(x), 1), 'Self random numbers histogram', 100)
#    distribution(randNum(x, n), 'Desc')
#    peaks(randNum(x, n), n)
#    vector(shift(x, 100), 'Desc')
#    vector(x, 'Desc')


if __name__ == "__main__":
    fig = plt.figure(1)
    N = 1000
    k = random.uniform(2, N)
    b = random.uniform(0, N)
    alp = random.random()
    bet = random.random()
    x = np.arange(0, N)
#    first_task(x, k, b, alp, bet)
#    second_task(x, N)
#    print(station(normNum(randNum(x, N), 1), 10))
    third_task(x, N)
#    print(statistics(normNum(randNum(x), 1)))

