import matplotlib.gridspec as gr
from inout.plot import *
from inout.fopen import *
from analysis.image_analysis import *
import numpy as np
from matplotlib import pyplot as plt
import cv2
from analysis import stats
from analysis import amplitude_modulation
from model import trend, random, shifts
from scipy import ndimage
from scipy.misc import imresize
from numba import jit
dpi = 120


def min_image_statistics_output(picture):
    # дисперсию, рседнее значение, минимум, максимум
    min = picture.min()
    max = picture.max()
    std = picture.std()
    #print(f"Min: {min}, Max: {max}, Std: {std}")
    rows = picture.shape[0]
    row_average = np.zeros(rows)
    row_variance = np.zeros(rows)
    for i in range(rows):
        row_average[i] = picture[i].mean()
        row_variance[i] = picture[i].var()
    #построчный вывод
    gs = gr.GridSpec(2, 2)
    plt.subplot(gs[0, 0])
    plot(row_variance, desc='Дисперсия по строкам')
    plt.subplot(gs[0, 1])
    plot(row_average, desc='Среднее по строкам')
    #вывод по столбцам
    columns = picture.shape[1]
    column_average = np.zeros(columns)
    column_variance = np.zeros(columns)
    for i in range(columns):
        column_average[i] = picture[:, i].mean()
        column_variance[i] = picture[:, i].var()
    plt.subplot(gs[1, 0])
    plot(column_variance, desc='Дисперсия по столбцам')
    plt.subplot(gs[1, 1])
    plot(column_average, desc='Среднее по столбцам')
    plt.show()


def brightness_output(picture):
    #нормализация
    #print(normalize(picture, 1))
    #гистограмма яркости
    plt.hist(picture.flatten(), 255)


def resize(picture, scale, method="nearest"):
    rows = picture.shape[0]
    columns = picture.shape[1]
    image = np.empty((int(rows * scale), int(columns * scale), 3), dtype=picture.dtype)

    if method not in ("nearest", "bilinear"):
        raise ValueError("unknown resampling filter")
    plt.figure(figsize=(image.shape[0] / dpi, image.shape[1] / dpi))
    if method == "nearest":
        plt.imshow(nearest_neighbour(image, picture, scale), aspect='auto')
        plt.title("Метод ближайшего соседа")
    if method == "bilinear":
        plt.imshow(bilinear(picture, image), aspect='auto')
        plt.title("Метод билинейной интерполяции")
#       plt.imshow(imresize(arr=picture, size=image.shape, interp="bilinear", mode="RGB"))


def negative(picture):
    plt.imshow(255 - picture, cmap='gray')


def gamma(picture, gamma=0.9, c=1):
    return c * (picture ** gamma)


def log(picture, c=1, base=0.9):
    _pic = np.array(picture, dtype=np.float)
    b = np.log(_pic+1, dtype=np.float)/np.log(base)
    return b


def equalisation(picture):
    hist, bins = np.histogram(picture.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max() * 255
    #print(cdf_normalized.max(), len(cdf_normalized))
    rows = picture.shape[0]
    columns = picture.shape[1]
    new_pic = np.empty(picture.shape)
    for i in range(rows):
        for j in range(columns):
            new_pic[i,j] = cdf_normalized[int(float(picture[i][j]))]
    plt.imshow(new_pic, cmap='gray')
    plt.show()
    plt.hist(new_pic.flatten(), color='r')
    plt.hist(picture.flatten(), color='b')
    plt.show()


def lsFilters(picture):
    threshimg = np.zeros_like(picture)
    w = picture > 200
    b = picture < 200
    threshimg[w], threshimg[b] = 255, 0
    # Фильтры Собеля
    maskh = np.array([[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]])
    maskv = maskh.T
    # Лапласиан
    laplas = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

    padded = np.pad(threshimg, (1, 1), 'constant')
    vert = ndimage.convolve(padded, maskv)  # vertical edges
    horiz = ndimage.convolve(padded, maskh)  # horizontal edges
    plt.title('Vertical')
    plt.imshow(vert, cmap='gray')

    plt.figure()
    plt.title('Horizontal')
    plt.imshow(horiz, cmap='gray')
    laplasian = ndimage.convolve(padded, laplas)
    plt.figure()
    plt.title('Lablasian')
    plt.imshow(laplasian, cmap='gray')


def filter(val):
    gs = gr.GridSpec(1, 1)
    ft5 = stats.fourier_transform(val[5], len(val))
    ft100 = stats.fourier_transform(val[100], len(val))[1][1:]
    corr = np.zeros(len(ft100))
    for i in range(0, len(ft100)):
        corr[i] = stats.correlation(ft100, ft100, i)
    plt.subplot(gs[0, 0])
    corr_x = np.arange(0, 0.5, 0.5 / len(corr // 2))
    plt.plot(corr_x, corr)
    #filter
    bpF = amplitude_modulation.BSF(0.2, 0.4, 1, 16)
    bpF_conv = np.zeros(val.shape)
    for i in range(0, len(val)):
        bpF_conv[i] = trend.convolution(val[i], bpF)
    return bpF_conv


def noises(picture):
    gs = gr.GridSpec(1, 1)
    #histogram
    plt.subplot(gs[0, 0])
    brightness_output(picture)
    plt.show()
    #noise
    gs = gr.GridSpec(1, 2)
    plt.subplot(gs[0, 0])
    plt.imshow(rnd_noise(picture), cmap='gray')
    plt.title('Случайный шум')
    plt.subplot(gs[0, 1])
    plt.imshow(normalize(salt_pepper(picture), 255), cmap='gray')
    plt.title('Соль-перец')


@jit
def rnd_noise(picture):
    noise = np.empty(picture.shape)
    for i in range(len(picture)):
        noise[i] = random.rnd(0, 20, n=picture.shape[1]) + picture[i]
    return noise


@jit
def salt_pepper(picture):
    noise = np.empty(picture.shape)
    for i in range(len(picture)):
        noise[i] = shifts.peaks(0.05, 80, np.zeros(picture.shape[1])) + picture[i]
    return noise


@jit
def median_filter(data, mask=3):
    indexer = mask // 2
    data_final = np.empty(data.shape)
    for i in range(len(data)):
        inx_st = 0 if i - indexer < 0 else i - indexer
        inx_end = data.shape[0] if i + indexer > data.shape[0] else i + indexer
        for j in range(len(data[0])):
            jnx_st = 0 if j - indexer < 0 else j - indexer
            jnx_end = data.shape[1] if j + indexer > data.shape[1] else j + indexer
            data_final[i, j] = np.median(data[inx_st:inx_end, jnx_st:jnx_end])
    return data_final


@jit
def average_filter(data, mask=3):
    indexer = mask // 2
    data_final = np.empty(data.shape)
    for i in range(len(data)):
        inx_st = 0 if i - indexer < 0 else i - indexer
        inx_end = data.shape[0] if i + indexer > data.shape[0] else i + indexer
        for j in range(len(data[0])):
            jnx_st = 0 if j - indexer < 0 else j - indexer
            jnx_end = data.shape[1] if j + indexer > data.shape[1] else j + indexer
            data_final[i, j] = data[inx_st:inx_end, jnx_st:jnx_end].mean()
    return data_final


@jit
def image_restoration(values):
    gs = gr.GridSpec(1, 2)
    plt.subplot(gs[0, 0])
    ft = np.zeros(values.shape)
    for i in range(values.shape[0]):
        ft[i] = stats.fourier_transform(values[i], values.shape[1])[0]
    for i in range(values.shape[1]):
        ft[:, i] = stats.fourier_transform(ft[:, i], values.shape[0])[0]

    plt.imshow(ft, cmap='gray')
    plt.title("Прямое Фурье")
    plt.subplot(gs[0, 1])

    for i in range(values.shape[0]):
        ft[i] = stats.inverse_fourier_transform(ft[i], values.shape[1])
    for i in range(values.shape[1]):
        ft[:, i] = stats.inverse_fourier_transform(ft[:, i], values.shape[0])

    plt.imshow(ft, cmap='gray')
    plt.title("Обратное Фурье")


def result():
    scale = 4
    plt.figure()
    ##xcr изображения
    #xcr = xcr_values("data/h400x300.xcr")
    # picture = img_values("data/grace.jpg")
    # plt.imshow(picture, cmap='gray')
    # plt.figure()
    # min_image_statistics_output(picture)
    # brightness_output(picture)

    ##Grace Kelly изменение размера
    # picture = img_values("data/grace.jpg")
    # plt.imshow(picture, cmap='gray')
    # plt.figure()
    # resize(picture, scale, method="nearest")
    # resize(picture, scale, method="bilinear")

    ##Grace Kelly изменение вида изображения
    ##гамму сделать меньше и больше 1, с > 0
    # picture = to_one_channel(img_values("data/image2.jpg"))
    # gs = gr.GridSpec(2, 2)
    # plt.subplot(gs[0, 0])
    # plt.imshow(picture, cmap='gray', interpolation="none")
    # plt.subplot(gs[0, 1])
    # negative(picture)
    # plt.title('Негатив')
    # plt.subplot(gs[1, 0])
    # pic = np.array(normalize(gamma(picture, 0.5)), dtype=np.int)
    # plt.imshow(pic, cmap='gray', interpolation="none")
    # plt.title('Гамма коррекция')
    # plt.subplot(gs[1, 1])
    # pic2 = np.array(normalize(log(picture, 5, base=1.11)), dtype=np.int)
    # plt.imshow(pic2, cmap='gray', interpolation="none")
    # plt.title('Логарифмическая коррекция')
    # plt.show()

    ##Эквализация и приведение
    # image = img_values("data/HollywoodLC.jpg")
    # # image = to_one_channel(img_values("data/MODEL.jpg"))
    # plt.imshow(image, cmap='gray')
    # plt.figure()
    # equalisation(image)

    ##Фильтр частот
    # xcr = xcr_values("data/h400x300.xcr")
    # plt.imshow(xcr, cmap='gray')
    # plt.figure()
    # plt.imshow(filter(xcr), cmap='gray')

    ##Шумы
    picture = to_one_channel(img_values("data/MODEL.jpg"))
    noises(picture)
    plt.figure()
    gs = gr.GridSpec(2, 2)
    plt.subplot(gs[0, 0])
    plt.imshow(median_filter(normalize(rnd_noise(picture), 5)), cmap='gray')
    plt.title('Медианный фильтр')
    plt.subplot(gs[0, 1])
    plt.imshow(average_filter(normalize(rnd_noise(picture), 5)), cmap='gray')
    plt.title('Усредненный фильтр')
    plt.subplot(gs[1, 0])
    plt.imshow(median_filter(normalize(salt_pepper(picture), 5)), cmap='gray')
    plt.subplot(gs[1, 1])
    plt.imshow(average_filter(normalize(salt_pepper(picture), 5)), cmap='gray')

    #Фурье
    # picture = to_one_channel(img_values("data/image2.jpg"))
    # image_restoration(picture)









