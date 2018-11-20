import numpy as np
from model.shifts import peaks
import math
import random



#исправить
def expon(a, x):
    return np.exp(a * x)


def linear(k, x, b):
    return k * x + b


def cardiography(m, a=25, t=0.005, f=10):
    k = np.arange(m)
    temp = np.sin(2 * np.pi * f * t * k) * expon(-a, k * t)
    return temp / temp.max()


def heartbeat(n=1000, m=200):
    x = np.zeros(n)
    for i in np.arange(m, n, m):
        x[i] = np.random.uniform(m - 90, m - 70)
    return x


def convolution(x, h):
    y = np.zeros(len(x) + len(h))
    for k in range(len(x)):
        for i in range(len(h)):
            y[k] += x[k - i] * h[i]
    return y[:-len(h)]


def low_pass_filter(m=128, fcut=15, dt=0.002):
    const = [0.35577019, 0.23469830, 0.07211497, 0.00630165]
    lpW = np.zeros(m + 1)
    arg = 2 * fcut * dt
    lpW[0] = arg
    arg *= np.pi
    for i in range(1, m + 1):
        lpW[i] = np.sin(arg * i) / (np.pi * i)
    lpW[m] /= 2
    sumg = lpW[0]
    for i in range(1, m + 1):
        sum = const[0]
        arg = np.pi * i / m
        for k in range(1, 4):
            sum += 2 * const[k] * np.cos(arg * k)
        lpW[i] *= sum
        sumg += 2 * lpW[i]
    lpW /= sumg
    return np.append(lpW[::-1], lpW[1:]) # * 2 * m
