import random
import time


def normNum(x, s):
    return [(((xx - min(x))/(max(x) - min(x))) - 0.5) * 2 * s for xx in x]


def randNum(x):
    return [random.random() for xx in x]


def selfRand(x):
    return [hash(time.clock() / (1 + xx)) for xx in x]

