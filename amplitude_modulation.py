import numpy as np


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

