import numpy as np
import math
from numba import jit


def signal(n=1000, amplitude=100):
    _x = np.arange(n)
    for i in range(n):
        _x[i] = amplitude * np.sin(i / (10 * np.pi))
    return _x


def modulation(m=0.5, amplitude=100, frequency=57, phase=1):
    _x = signal()
    _modul = np.arange(len(_x))
    _max = np.abs(np.max(_x))
    for i in range(len(_x)):
        _modul[i] = amplitude * (1 + m * _x[i] / _max) * np.cos(frequency * i * phase)
    return _modul


def carrying_oscillation(m=0.5, amplitude=100, frequency=57):
    _x = signal()
    _modul = np.arange(len(_x))
    for i in range(len(_x)):
        _modul[i] = np.abs(amplitude + m) * np.cos(frequency * i)
    return _modul


def LPF(Fcut: float, dT: float, m: int = 32) -> list:
    d = np.array(
		[0.35577019,
	     0.2436983,
	     0.07211497,
	     0.00630165],
	dtype=np.double)
    idxs = np.array([i for i in range(1, m + 1)])

    # Create array
    lpw = np.array([0] * (m + 1), dtype=np.double)

    # Some magic here
    arg = 2 * Fcut * dT
    lpw[0] = arg
    arg *= np.pi
    lpw[1:] = np.divide(
		np.sin(np.multiply(idxs, arg)),
		np.multiply(idxs, np.pi))

# make trapezoid:
    lpw[-1] /= 2
    sumg = lpw[0]
    for i in range(1, m + 1):
        _sum = d[0]
        arg = math.pi * i / m
        for k in range(1, 4):
            _sum += 2 * d[k] * math.cos(arg * k)
        lpw[i] *= _sum
        sumg += 2 * lpw[i]
    # normalization
    lpw = np.divide(lpw, sumg)

    answer = lpw[::-1].tolist()
    answer.extend(lpw[1:].tolist())
    return answer


@jit
def HPF(Fcut: float, dT: float, m: int = 32) -> list:
    lpw = np.array(LPF(Fcut=Fcut, dT=dT, m=m))

    lpw = np.multiply(lpw, -1)

    lpw[m] = 1 + lpw[m]

    return lpw.tolist()


@jit
def BPF(Fcut1: float, Fcut2: float, dT: float, m: int = 32) -> list:
    lpw1 = np.array(LPF(Fcut=Fcut1, dT=dT, m=m))
    lpw2 = np.array(LPF(Fcut=Fcut2, dT=dT, m=m))
    lpw = np.subtract(lpw2, lpw1)
    return lpw.tolist()


def BSF(Fcut1: float, Fcut2: float, dT: float, m: int = 32) -> list:
    lpw1 = np.array(LPF(Fcut=Fcut1, dT=dT, m=m))
    lpw2 = np.array(LPF(Fcut=Fcut2, dT=dT, m=m))
    lpw = np.subtract(lpw1, lpw2)
    lpw[m] += 1
    return lpw.tolist()


def anti_trend_(arr, window_width: int = None) -> list:
    trend = []
    if window_width is None:
        window_width = int(len(arr) / 100)

    counter = int(math.floor(len(arr) / window_width))
    for i in range(counter):
        mean_v = arr[i * window_width: (i + 1) * window_width].mean()
        trend.append(mean_v)
        for x in range(window_width):
            arr[i * window_width + x] -= mean_v

    if window_width * counter != len(arr):
            mean_v = arr[window_width * counter:].mean()
            trend.append(mean_v)
            for x in range(window_width * counter, len(arr)):
                arr[x] -= mean_v
    return trend


def anti_spike(arr, K: int = 4) -> None:
    avg = arr.mean()
    sigma = arr.std()
    for x in range(len(arr)):
        if math.fabs(arr[x]) > avg + K * sigma:
            if 0 < x < len(arr) - 1:
                arr[x] = (arr[x + 1] + arr[x + 2]) / 2
            elif x == len(arr) - 1:
                arr[x] = (arr[x - 2] + arr[x - 1]) / 2
            else:
                arr[x] = (arr[x-1] + arr[x + 1]) / 2