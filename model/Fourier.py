from analysis import stats
import numpy as np
from numba import jit


@jit
def ft(values):
    ft = np.zeros(values.shape)
    for i in range(values.shape[0]):
        ft[i] = stats.fourier_transform(values[i])[0]
    for i in range(values.shape[1]):
        ft[:, i] = stats.fourier_transform(ft[:, i])[0]
    return ft


@jit
def inverse_ft(ft):
    for i in range(ft.shape[0]):
        ft[i] = stats.inverse_fourier_transform(ft[i])
    for i in range(ft.shape[1]):
        ft[:, i] = stats.inverse_fourier_transform(ft[:, i])
    return ft