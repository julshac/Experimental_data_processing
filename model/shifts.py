import random


def peaks(probability, amplitude, x):
    for i in range(len(x)):
        if random.uniform(0, 1) < probability:
            if random.uniform(0, 1) < 0.5:
                x[i] = -1 * random.uniform(1, amplitude)
            else:
                x[i] = 1 * random.uniform(1, amplitude)
    return x


def shift(x, const):
    return x + const
