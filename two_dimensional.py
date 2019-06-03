import matplotlib.gridspec as gr
from inout.plot import *
from inout.fopen import *
from analysis.image_analysis import *
import numpy as np
from matplotlib import pyplot as plt
from model import Fourier
from analysis import stats
from analysis import amplitude_modulation
from model import trend, random, shifts, _fourier
import model._fourier as m
from scipy import ndimage
from numba import jit
import cv2

dpi = 120


def min_image_statistics_output(picture):
    # дисперсию, рседнее значение, минимум, максимум
    min = picture.min()
    max = picture.max()
    std = picture.std()
    # print(f"Min: {min}, Max: {max}, Std: {std}")
    rows = picture.shape[0]
    row_average = np.zeros(rows)
    row_variance = np.zeros(rows)
    for i in range(rows):
        row_average[i] = picture[i].mean()
        row_variance[i] = picture[i].var()
    # построчный вывод
    gs = gr.GridSpec(2, 2)
    plt.subplot(gs[0, 0])
    plot(row_variance, desc='Дисперсия по строкам')
    plt.subplot(gs[0, 1])
    plot(row_average, desc='Среднее по строкам')
    # вывод по столбцам
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
    # нормализация
    # print(normalize(picture, 1))
    # гистограмма яркости
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
    b = np.log(_pic + 1, dtype=np.float) / np.log(base)
    return b


def equalisation(picture):
    hist, bins = np.histogram(picture.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max() * 255
    # print(cdf_normalized.max(), len(cdf_normalized))
    rows = picture.shape[0]
    columns = picture.shape[1]
    new_pic = np.empty(picture.shape)
    for i in range(rows):
        for j in range(columns):
            new_pic[i, j] = cdf_normalized[int(float(picture[i][j]))]
    plt.imshow(new_pic, cmap='gray')
    plt.show()
    plt.hist(new_pic.flatten(), color='r')
    plt.hist(picture.flatten(), color='b')
    plt.show()


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
    # filter
    bpF = amplitude_modulation.BSF(0.2, 0.4, 1, 16)
    bpF_conv = np.zeros(val.shape)
    for i in range(0, len(val)):
        bpF_conv[i] = trend.convolution(val[i], bpF)
    return bpF_conv


def noises(picture):
    gs = gr.GridSpec(1, 1)
    # histogram
    plt.subplot(gs[0, 0])
    brightness_output(picture)
    plt.show()
    # noise
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


def image_distortion(values):
    gs = gr.GridSpec(1, 2)
    plt.subplot(gs[0, 0])
    values = np.array(values.real, dtype=np.float)
    plt.title("Исходное изображение")
    plt.imshow(values, cmap='gray')

    # ft = Fourier.ft(values)
    # ft = np.fft.rfft2(values)
    ft = m.fourier_2D(values, complex=True)

    data = dat_values("data/kernD76_f4.dat")

    # ft_dat = m.fourier_2D(data, complex=True)
    data = np.append(data, np.full(ft.shape[1] - len(data), 0))
    # ft_dat = np.fft.rfft2(data)
    # ft_dat = stats.fourier_transform(data)[0]
    ft_dat = m.fourier_complex(data)

    # ft_dat = np.append(ft_dat, np.full(ft.shape[1] - len(ft_dat), 1))

    for i in range(ft.shape[0]):
        ft[i] = ft[i] / ft_dat

    # ft = np.fft.irfft2(ft)
    ft = m.fourier_2D_back(ft, complex=True)
    # ft = Fourier.inverse_ft(ft)
    ft = np.array(ft.real, dtype=np.float)

    plt.subplot(gs[0, 1])
    plt.title("Искаженное изображение")
    plt.imshow(ft, cmap='gray')


def image_distortion_noise(values):
    gs = gr.GridSpec(1, 2)
    plt.subplot(gs[0, 0])
    values = np.array(values.real, dtype=np.float)
    plt.title("Исходное изображение")
    plt.imshow(values, cmap='gray')

    # ft = Fourier.ft(values)
    # ft = np.fft.rfft2(values)
    ft = m.fourier_2D(values, complex=True)

    data = dat_values("data/kernD76_f4.dat")
    # ft_dat = m.fourier_2D(data, complex=True)
    data = np.append(data, np.full(ft.shape[1] - len(data), 0))
    # ft_dat = np.fft.rfft2(data)
    # ft_dat = stats.fourier_transform(data)[0]
    ft_dat = m.fourier_complex(data)
    result = ft * ((ft_dat.conj()) / ((np.absolute(ft_dat) ** 2) + 0.00001))
    # ft_dat = np.append(ft_dat, np.full(ft.shape[1] - len(ft_dat), 1))

    # ft = np.fft.irfft2(ft)
    ft = m.fourier_2D_back(result, complex=True)
    # ft = Fourier.inverse_ft(ft)
    ft = np.array(ft.real, dtype=np.float)
    plt.subplot(gs[0, 1])
    plt.title("Искаженное изображение")
    plt.imshow(ft, cmap='gray')


def scaling2D(image, min, max, scale):
    res = []
    for i in image:
        row = []
        for j in i:
            f = (j - min) / (max - min) * scale
            row.append(int(f))
        res.append(row)
    return np.array(res)


def edge_restore(picture, img, treshold=165):
    restore = img
    if filter == "lpf":
        img = img.reshape(picture.shape)
        restore = picture - img
    restore = scaling2D(restore, min(restore.flat), max(restore.flat), 255)

    filtered = np.empty(0)
    for i in restore.flatten():
        if i > treshold:
            filtered = np.append(filtered, 255)
        else:
            filtered = np.append(filtered, 0)
    restore = filtered.reshape(300, 400)
    return restore


@jit
def edge_detection(picture, filter="lpf"):
    img = np.empty(picture.shape)
    if filter not in ("lpf", "hpf"):
        raise ValueError("unknown filter")
    if filter == "lpf":
        lpF = amplitude_modulation.LPF(0.05, dT=1, m=8)
        for i in range(picture.shape[0]):
            img[i] = trend.convolution(picture[i], lpF)
        for i in range(picture.shape[1]):
            img[:, i] = trend.convolution(img[:, i], lpF)
        plt.imshow(np.array(picture - img > 10, dtype=np.float) * 255, cmap='gray')
        # plt.imshow(edge_restore(picture, img), cmap='gray')
        plt.title("LPF")
    if filter == "hpf":
        hpF = amplitude_modulation.HPF(0.05, dT=1, m=8)
        for i in range(picture.shape[0]):
            img[i] = trend.convolution(picture[i], hpF)
        for i in range(picture.shape[1]):
            img[:, i] = trend.convolution(img[:, i], hpF)
        plt.imshow(np.array(img > 8, dtype=np.float) * 255, cmap='gray')
        # plt.imshow(edge_restore(picture, img), cmap='gray')
        plt.title("HPF")


@jit
def gradientLaplasian(picture):
    threshimg = np.zeros_like(picture)
    w = picture > 200
    b = picture < 200
    threshimg[w], threshimg[b] = 255, 0
    # маска Собеля (взять левую маску, правую, взять от них модуль и сложить)
    mask_sobel = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])
    mask = mask_sobel.T
    # Лапласиан
    laplas = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

    padded = np.pad(threshimg, (1, 1), 'constant')
    vert = ndimage.convolve(padded, mask)  # вертикальная граница
    hor = ndimage.convolve(padded, mask_sobel)  # горизонтальная граница
    gs = gr.GridSpec(2, 2)
    plt.subplot(gs[0, :])
    plt.title('Лапласиан')
    laplas = ndimage.convolve(padded, laplas)
    plt.imshow(laplas, cmap='gray')

    plt.subplot(gs[1, 0])
    plt.title('Вертикальные грани')
    plt.imshow(vert, cmap='gray')

    plt.subplot(gs[1, 1])
    plt.title('Горизонтальные грани')
    plt.imshow(hor, cmap='gray')


def erosionDilation(picture):
    # эрозия с определенной маской делает изображение меньше
    threshimg = np.zeros_like(picture)
    kernel = np.ones((5, 5), np.uint8)  # Морфологические образы
    w = picture > 200
    b = picture < 200
    threshimg[w], threshimg[b] = 255, 0

    gs = gr.GridSpec(3, 2)
    plt.subplot(gs[0, :])
    plt.title('Оригинальное пороговое изображение')
    plt.imshow(threshimg, cmap='gray')

    plt.subplot(gs[1, 0])
    erosion = cv2.erode(threshimg, kernel, iterations=1)
    plt.title('Эрозия')
    plt.imshow(erosion, cmap='gray')

    plt.subplot(gs[1, 1])
    dilation = cv2.dilate(threshimg, kernel, iterations=1)
    plt.title("Дилатация")
    plt.imshow(dilation, cmap='gray')

    plt.subplot(gs[2, 0])
    plt.title('Выделение контура (дил - порог)')
    plt.imshow(dilation - threshimg, cmap='gray')

    contour = dilation - erosion
    plt.subplot(gs[2, 1])
    plt.title('Выделение контура (дил - эр)')
    plt.imshow(contour, cmap='gray')

    dilation2 = cv2.dilate(erosion, kernel, iterations=1)
    plt.figure()
    plt.imshow(dilation2, cmap='gray')


# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import argparse
import imutils


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def stonesSearch(picture, width=6):
    # threshimg = np.zeros_like(picture)
    # kernel = np.ones((5, 5), np.uint8)  # Морфологические образы
    # w = picture > 130
    # b = picture < 90
    # threshimg[w], threshimg[b] = 255, 0
    #
    # gs = gr.GridSpec(1, 1)
    # plt.subplot(gs[0, :])
    # plt.title('Оригинальное пороговое изображение')
    # plt.imshow(threshimg, cmap='gray')

    # load the image, convert it to grayscale, and blur it slightly
    image = cv2.imread("data/stones.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None
    # loop over the contours individually
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 10:
            continue

        # compute the rotated bounding box of the contour
        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

        # loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # draw the midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                 (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                 (255, 0, 255), 2)

        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # if the pixels per metric has not been initialized, then
        # compute it as the ratio of pixels to supplied metric
        # (in this case, inches)
        if pixelsPerMetric is None:
            pixelsPerMetric = dB / width

        # compute the size of the object
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        # draw the object sizes on the image
        cv2.putText(orig, "{:.1f}in".format(dimA),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        cv2.putText(orig, "{:.1f}in".format(dimB),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)

        # show the output image
        cv2.imshow("Image", orig)
        cv2.waitKey(0)


def result():
    scale = 4
    plt.figure()
    ##xcr изображения
    # xcr = xcr_values("data/h400x300.xcr")
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
    # picture = to_one_channel(img_values("data/MODEL.jpg"))
    # noises(picture)
    # plt.figure()
    # gs = gr.GridSpec(2, 2)
    # plt.subplot(gs[0, 0])
    # plt.imshow(median_filter(normalize(rnd_noise(picture), 5)), cmap='gray')
    # plt.title('Медианный фильтр')
    # plt.subplot(gs[0, 1])
    # plt.imshow(average_filter(normalize(rnd_noise(picture), 5)), cmap='gray')
    # plt.title('Усредненный фильтр')
    # plt.subplot(gs[1, 0])
    # plt.imshow(median_filter(normalize(salt_pepper(picture), 5)), cmap='gray')
    # plt.subplot(gs[1, 1])
    # plt.imshow(average_filter(normalize(salt_pepper(picture), 5)), cmap='gray')

    # 2D Фурье
    # picture = to_one_channel(img_values("data/image2.jpg"))
    # image_restoration(picture)

    # Искажение изображения. Выделение 3ки
    # picture = dat_2D_reader("data/blur307x221D.dat")
    # image_distortion_noise(picture)
    # plt.show()
    # picture = dat_2D_reader("data/blur307x221D_N.dat")
    # image_distortion_noise(rnd_noise(picture))
    # plt.show()
    # image_distortion_noise(salt_pepper(picture))

    # выделение границ
    # picture = to_one_channel(img_values("data/MODEL.jpg"))
    # plt.imshow(picture, cmap='gray')
    # plt.title('Исходное изображение')
    # gs = gr.GridSpec(3, 3)
    # plt.subplot(gs[0, 0])
    # plt.imshow(picture, cmap='gray')
    # plt.title('Исходное изображение')
    # plt.subplot(gs[1, 0])
    # edge_detection(picture, "lpf")
    # plt.subplot(gs[2, 0])
    # edge_detection(picture, "hpf")
    # plt.subplot(gs[0, 1])
    # rnd = rnd_noise(picture)
    # plt.imshow(rnd, cmap='gray')
    # plt.title('Случайный шум')
    # plt.subplot(gs[1, 1])
    # edge_detection(rnd, "lpf")
    # plt.subplot(gs[2, 1])
    # edge_detection(rnd, "hpf")
    # plt.subplot(gs[0, 2])
    # sp = salt_pepper(picture)
    # plt.imshow(salt_pepper(picture), cmap='gray')
    # plt.title('Соль-перец')
    # plt.subplot(gs[1, 2])
    # edge_detection(sp, "lpf")
    # plt.subplot(gs[2, 2])
    # edge_detection(sp, "hpf")

    # выделение границ (Лапласиан, гралиент (Собель))
    # picture = to_one_channel(img_values("data/MODEL.jpg"))
    # plt.imshow(picture, cmap='gray')
    # plt.title('Исходное изображение')
    # plt.figure()
    # gradientLaplasian(picture)
    # plt.show()
    # gradientLaplasian(rnd_noise(picture))
    # plt.show()
    # gradientLaplasian(salt_pepper(picture))

    # #эрозия и дилатация
    # picture = to_one_channel(img_values("data/MODEL.jpg"))
    # erosionDilation(picture)
    # plt.show()
    # erosionDilation(rnd_noise(picture))
    # plt.show()
    # erosionDilation(salt_pepper(picture))

    # Поиск камня размера n. Размер камня = 6
    picture = to_one_channel(img_values("data/stones.jpg"))
    stonesSearch(picture, 0.006)
