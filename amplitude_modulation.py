import numpy as np


def signal(n=1000, amplitude=100):
    x = np.arange(n)
    for i in range(n):
        x[i] = amplitude * np.cos(i / (10 * np.pi))
    return x


def modulation(m=0.5, amplitude=100, frequency=57, phase=1):
    x = signal()
    modul = np.arange(len(x))
    max = np.abs(np.max(x))
    for i in range(len(x)):
        modul[i] = amplitude * (1 + m * x[i] / max) * np.cos(frequency * i * phase)
    return modul


def carrying_oscillation(m=0.5, amplitude=100, frequency=57):
    x = signal()
    modul = np.arange(len(x))
    for i in range(len(x)):
        modul[i] = np.abs(amplitude + m) * np.cos(frequency * i)
    return modul

