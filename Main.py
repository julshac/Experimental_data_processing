import matplotlib.pyplot as plt
import numpy as np
import random
from Graphics.Painting import vector, histogram, distribution
from Additions.Randoms import selfRand, randNum, normNum
from Additions.Shifts import peaks, shift
from Graphics.Figures import fig_1, fig_2, fig_3, fig_4


def first_task(x, k, b, alp, bet):
    vector(fig_1(x, k), 'y = kx')
    vector(fig_2(k, x, b), 'y = -kx + b')
    vector(fig_3(alp, x), 'y = e^ax')
    vector(fig_4(bet, x), 'y = e^-bx')


def second_task(x, n):
    vector(randNum(x, n), 'Random numbers')
    vector(selfRand(x), 'Self random numbers')
    vector(normNum(randNum(x, n), 1), 'Random numbers')
    vector(normNum(selfRand(x), 1), 'Self random numbers')
    distribution(normNum(randNum(x, n), 1), 'Random numbers trend')
    distribution(normNum(selfRand(x), 1), 'Random numbers trend')


def third_task(x, n):
    histogram(randNum(x, n), 'Desc')
    distribution(randNum(x, n), 'Desc')
    peaks(randNum(x, n), n)
    vector(shift(x, 100), 'Desc')
    vector(x, 'Desc')


if __name__ == "__main__":
    fig = plt.figure(1)
    N = 1000
    k = random.uniform(2, N)
    b = random.uniform(0, N)
    alp = random.uniform(0, 1)
    bet = random.uniform(0, 1)
    x = np.arange(0, N, 10)
    first_task(x, k, b, alp, bet)
    second_task(x, N)


