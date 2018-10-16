import numpy as np
import time


def normNum(x, s):
    return (((x - min(x))/(max(x) - min(x))) - 0.5) * 2 * s


def randNum(n):
    return np.random.sample(n)


def selfRand(N):
    x = np.random.randint(0, 100, N) #np.arange(0, N)
    #return np.log(hash(time.clock()), x)
    return x

