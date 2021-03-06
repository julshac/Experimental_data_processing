import numpy as np
import time
import datetime as dt


def normalize(numbers: np.array, normalization_bound: int) -> np.array:
    return (((numbers - min(numbers)) / (max(numbers) - min(numbers))) - 0.5) * 2 * normalization_bound


def numpy_random(count: int) -> np.array:
    return np.random.sample(count)


def rnd(low, high, n=1000):
    t = np.zeros(n)
    for i in range(n):
        t[i] = np.random.uniform(low, high)
    return t


def my_random(count) -> np.array:
    core = np.random.normal(size=(count, 1))
    return np.log(np.sin(hash(core)))

