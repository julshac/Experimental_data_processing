import random


def peaks(probability, amplitude):
    if random.uniform(0, 1) < probability:
        if random.uniform(0, 1) < 0.5:
            return -1 * random.uniform(1, amplitude)
        else:
            return 1 * random.uniform(1, amplitude)
    else:
        return 0


def shift(x, const):
    return [xx + const for xx in x]
