from numba import jit
import numpy as np


def normalize(f, s=255):
    return ((f - f.min()) / (f.max() - f.min())) * s


@jit
def nearest_neighbour(image, picture, scale):
    for i in range(image.shape[0] - scale):
        for j in range(image.shape[1] - scale):
            image[i][j] = picture[round(i / scale)][round(j / scale)]
    #обработка краев
    for i in range(image.shape[0] - scale, image.shape[0]):
        for j in range(image.shape[1] - scale, image.shape[1]):
            image[i][j] = picture[int(i / scale)][int(j / scale)]
    return image


@jit
def interpolation(image, x, y):
    result = []
    x1 = int(x)
    y1 = int(y)
    x2 = x - x1
    y2 = y - y1
    x1lim = min(x1 + 1, image.shape[1] - 1)
    y1lim = min(y1 + 1, image.shape[0] - 1)

    for angle in range(image.shape[2]):
        b1 = image[y1, x1, angle]
        b2 = image[y1, x1lim, angle]
        t1 = image[y1lim, x1, angle]
        t2 = image[y1lim, x1lim, angle]

        b = x2 * b2 + (1. - x2) * b1
        t = x2 * t2 + (1. - x2) * t1
        pxf = y2 * t + (1. - y2) * b
        result.append(int(pxf + 0.5))
    return result


@jit
def bilinear(picture, image):
    row_scale = float(picture.shape[0]) / float(image.shape[0])
    column_scale = float(picture.shape[1]) / float(image.shape[1])

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            row_position = i * row_scale
            col_position = j * column_scale
            image[i, j] = interpolation(picture, col_position, row_position)
    return image


def fourier(data):
    rows = data.shape[0]
    columns = data.shape[1]
    re = np.zeros((rows, columns))
    im = np.zeros((rows, columns))
    # x = list(map(float, x))
    for n in range(rows - 1):
        for m in range(columns - 1):
          re[n, m], im[n, m] = fourier_step(data, rows, columns)
    re /= rows * columns
    im /= rows * columns
    cs = (re + im)
    cn = (np.sqrt(re ** 2 + im ** 2))
    return cs, cn[:len(cn) // 2]


def fourier_step(data, row, col):
    re, im = 0, 0
    for n in range(row - 1):
        for m in range(col - 1):
            re += data[row][col] * np.cos((2 * np.pi * n * m) / row * col)
            im += data[row][col] * np.sin((2 * np.pi * n * m) / row * col)
    return re, im

'''def interpolation(image, x, y, x1, y1, x2, y2):
 return ((image[x1][y1] * (x2 - x) * (y2 - y)) / ((x2 - x1) * (y2 - y1)))\
           + ((image[x2][y1] * (x - x1) * (y2 - y)) / ((x2 - x1) * (y2 - y1)))\
           + ((image[x1][y2] * (x2 - x) * (y - y1)) / ((x2 - x1) * (y2 - y1)))\
           + ((image[x2][y2] * (x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1)))'''

'''def bilinear(image, scale):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            x = i // scale
            y = j // scale
            image[i][j] = interpolation(image, i, j, x, y, x + 1, y + 1)
    return image'''

