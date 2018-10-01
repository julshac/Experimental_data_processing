import random


def peaks(prob, sig):
    if random.uniform(0, 1) < prob:
        if random.uniform(0, 1) < 0.5:
            return -1 * sig
        else: return 1
    else: return 0


def shift(x, const):
    return [xx + const for xx in x]
