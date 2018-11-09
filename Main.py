import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.gridspec as gr
from inout.plot import plot, distribution
from inout.fopen import dat_values
from model.random import my_random, numpy_random, normalize
from model.shifts import peaks, shift, reverse_shift, remove_peaks
from model.trend import linear, expon
from analysis.probfun import sin
from analysis.stats import stationarity, statistics, correlation, harmonic_motion, fourier_transform,\
                           inverse_fourier_transform


def first_task(x, k, b, alp, bet):
    plot(linear(k, x, 0), desc='y = kx')
    plot(linear(-k, x, b), desc='y = -kx + b')
    plot(expon(alp, x), desc='y = e^ax')
    plot(expon(-bet, x), desc='y = e^-bx')


def second_task(N):
    plot(numpy_random(N), desc='Random numbers')
    plot(my_random(N), desc='Self random numbers')
    plot(normalize(numpy_random(N), 1), desc='Normalize random numbers')
    plot(normalize(my_random(N), 1), desc='Normalize self random numbers')


def third_task(x, rand):
    plot(shift(normalize(numpy_random(len(x)), 1), 1), desc='Shift for random')
    plot(shift(normalize(my_random(len(x)), 1), 1), desc='Shift for self random')
    plot(peaks(0.001, 5, np.zeros(len(x))), desc='Peaks for zero function')
    stationarity(rand, 10, 0.05)


def fourth_task(n):
    print('Random stats')
    statistics(normalize(numpy_random(n), 1))
    print('Self random stats')
    statistics(normalize(my_random(n), 1))
    distribution(normalize(numpy_random(n), 1), 'Random numbers histogram')
    distribution(normalize(my_random(n), 1), 'Self random numbers histogram')


def fifth_task(n):
    tmp = np.zeros(100)
    for i in range(0, 100):
        tmp[i] = correlation(normalize(numpy_random(n), 1), normalize(numpy_random(n), 1), i)
    plt.subplot(211)
    plot(tmp, desc='Взаимная корреляционная ф-ция')
    tmp = np.zeros(1000)
    for i in range(0, 1000):
        a = normalize(numpy_random(n), 1)
        tmp[i] = correlation(a, a, i)
    harm = harmonic_motion(x, 100, 3, 0.002)
    plt.subplot(212)
    plot(harm, desc='Гармонический процесс')
    plt.show()


def sixth_task(x, n):
    gs = gr.GridSpec(2, 2)
    harm = harmonic_motion(x, 100, 3, 0.002)
    plt.subplot(gs[0, :])
    plot(harm, desc='Гармонический процесс, 37 Гц')  # 3, 37, 137, 237, 337 [Гц]
    ft = fourier_transform(harm, n)
    plt.subplot(gs[1, 1])
    plot([xx * 0.5 for xx in range(0, len(ft[1]) // 2)], ft[1][:len(ft[1]) // 2], desc='Преобразование Фурье')
    ift = inverse_fourier_transform(ft[0], n)
    plt.subplot(gs[1, 0])
    plot(ift, desc='Обратное преобразование Фурье')
    plt.show()

    #данные из файла dat
    xarr = dat_values()
    xt = np.zeros(n)
    f1 = sin(15, 3)
    f2 = sin(100, 37)
    f3 = sin(25, 137)
    for (i, t) in zip(range(n), xarr):
        xt[i] = f1(t) + f2(t) + f3(t)
    plot(xt, desc='Полигармонический процесс')
    plt.show()
    ft = fourier_transform(xarr, n)
    harm = harmonic_motion(xarr, 100, 37, 0.002)
    plot(harm, desc='Процесс на основе данных .dat файла')  # 3, 37, 137, 237, 337 [Гц]
    plt.show()
    plt.subplot(211)
    # ft = fourier_transform(xarr, n)
    plot([xx * 0.5 for xx in range(0, len(ft[1]) // 2)], ft[1][:len(ft[1]) // 2], desc='Преобразование Фурье')
    plt.subplot(212)
    ift = inverse_fourier_transform(ft[0], N)
    plot([xx * 0.5 for xx in range(0, len(ift) // 2)], ift[:len(ift) // 2], desc='Обратное преобразование Фурье')
    plt.show()


def seventh_task(x, n):
    harmonic_autocorrelation = np.zeros(n)
    linear_autocorrelation = np.zeros(n)
    exponential_autocorrelation = np.zeros(n)
    harm = harmonic_motion(x, 100, 37, 0.002)
    for i in range(n):
        harmonic_autocorrelation[i] = correlation(harm, harm, i)
        linear_autocorrelation[i] = correlation(linear(1.3, x, 10000), linear(1.3, x, 10000), i)
        exponential_autocorrelation[i] = correlation(expon(0.0016, x), expon(0.0016, x), i)
    plt.subplot(211)
    plot(harmonic_autocorrelation, desc='Автокорреляция гармонического процесса')
    plt.subplot(212)
    distribution(harm, desc='Плотность вероятности гармонического процесса')
    plt.show()
    ft = fourier_transform(shift(my_random(n), 0), n)
    plt.subplot(211)
    plot(ft[1], desc='Преобразование Фурье для случайного набора')
    ift = inverse_fourier_transform(ft[0], N)
    plt.subplot(212)
    plot(ift, desc='Обратное преобразование Фурье для случайного набора')
    # нет зависимостей от количества пиков
    plt.show()
    # бесокнечный набор синусов и косинусов даст 1.
    plt.subplot(221)
    distribution(linear(1.3, x, 10000), desc='Плотность вероятности линейной ф-ции')
    plt.subplot(222)
    distribution(expon(0.0016, x), desc='Плотность вероятности экспоненциальной ф-ции')
    plt.subplot(223)
    plot(linear_autocorrelation, desc='Автокорреляция линейной ф-ции')
    plt.subplot(224)
    plot(exponential_autocorrelation, desc='Автокорреляция экспоненциальной ф-ции')
    plt.show()


def eight_task(x):
    harm = harmonic_motion(x, 100, 37, 0.002)
    xk = np.zeros(len(harm))
    plt.subplot(221)
    plot(shift(harm, 100), desc='Shift')
    plt.subplot(222)
    plot(reverse_shift(harm), desc='Anti-shift')
    plt.subplot(223)
    plot(peaks(0.01, 5, xk), desc='Peaks for zero function')
    plt.subplot(224)
    plot(remove_peaks(xk), desc='Remove peaks for zero function')
    plt.show()


def nine_task(x, n):
    gs = gr.GridSpec(3, 3)
    m = [1, 2, 3, 10, 50, 100]
    harm = harmonic_motion(x, 10, 5, 0.002)
    noise = normalize(numpy_random(n), 2)
    plt.subplot(gs[0, :])
    plot(harm, desc='Гармонический сигнал')
    plt.subplot(gs[1, :])
    plot(noise, desc='Шумы')
    plt.subplot(gs[2, :])
    plot(noise + harm, desc='Гармонический сигнал с шумами')
    plt.show()
    realisations = np.zeros((6, n))
    for index, i in enumerate(m):
        realisations[index] = (noise + harm) * i
    plt.subplot(gs[0, 0])
    plot(realisations[0], desc='M = 1')
    plt.subplot(gs[0, 1])
    plot(realisations[1], desc='M = 2')
    plt.subplot(gs[0, 2])
    plot(realisations[2], desc='M = 3')
    plt.subplot(gs[1, 0])
    plot(realisations[3], desc='M = 10')
    plt.subplot(gs[1, 1])
    plot(realisations[4], desc='M = 50')
    plt.subplot(gs[1, 2])
    plot(realisations[5], desc='M = 100')
    plt.subplot(gs[2, :])
    plot(realisations, desc='All')
    plt.show()


if __name__ == "__main__":
    fig = plt.figure(1, figsize=(12, 4), dpi=80)
    N = 1000
    k = random.uniform(2, N)
    b = random.uniform(1, N)
    alp = random.random()
    bet = random.random()
    x = np.arange(0, N)
    # first_task(x, 1.3, 1000, 0.0016, 6)
    # second_task(N)
    # rnd = randNum(N)
    # third_task(x, rnd)
    # fourth_task(N)
    # fifth_task(N)
    # sixth_task(x, N)
    # seventh_task(x, N)
    # eight_task(x)
    nine_task(x, N)
