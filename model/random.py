import numpy as np
import time


def normalize(numbers: np.array, normalization_bound: int) -> np.array:
    return (((numbers - min(numbers)) / (max(numbers) - min(numbers))) - 0.5) * 2 * normalization_bound


def numpy_random(count: int) -> np.array:
    return np.random.sample(count)


def my_random(count) -> np.array:
    x = np.zeros(count)
    for i in range(count):
        x[i] = np.log(hash(time.clock())) * np.sin(i)
    return x

