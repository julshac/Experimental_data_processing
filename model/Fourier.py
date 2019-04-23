from analysis import stats
import numpy as np
from numba import jit
from model import trend


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


def img_convolution(picture, filter):
    img = np.empty(picture.shape)
    for i in range(picture.shape[0]):
        img[i] = trend.convolution(picture[i], filter)
    for i in range(picture.shape[1]):
        img[:, i] = trend.convolution(img[:, i], filter)
    return img