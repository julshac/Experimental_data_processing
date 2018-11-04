import numpy as np
import random


def peaks(probability, amplitude, x):
    for i in range(len(x)):
        if random.uniform(0, 1) < probability:
            if random.uniform(0, 1) < 0.5:
                x[i] = -1 * random.uniform(1, amplitude)
            else:
                x[i] = 1 * random.uniform(1, amplitude)
    return x


def remove_peaks(x):
    signal = np.copy(x)
    mean = signal.mean()
    std = signal.std()
    signal = (signal - mean) / std
    for i in range(1, signal.shape[0] - 1):
        if abs(signal[i]) > 2:
            signal[i] = (signal[i - 1] + signal[i + 1]) / 2
    signal = (signal * std) + mean
    return signal


def shift(x, const):
    return x + const


def reverse_shift(x):
    return x - x.mean()
