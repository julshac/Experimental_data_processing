import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.gridspec as gr
from inout.plot import *
from inout.fopen import *
from model.random import *
from model.shifts import *
from model.trend import *
from analysis.probfun import *
from analysis.stats import *
from amplitude_modulation import *


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


def nine_task(n):
    gs = gr.GridSpec(3, 1)
    m = [1, 2, 3, 10, 50, 100]
    s = 100
    harm = harmonic_motion(n, 5, s/20, 0.002)
    noise = normalize(numpy_random(n), 2)
    exp_noise = expon(0.0016, x) + noise
    plt.subplot(gs[0, :])
    plot(exp_noise, desc="trend")
    plt.subplot(gs[1, :])
    plot(anti_trend(exp_noise, noise), desc="Anti-trend")
    plt.subplot(gs[2, :])
    plot(noise, desc="Шум")
    plt.show()
    gs = gr.GridSpec(3, 3)
    plt.subplot(gs[0, :])
    plot(harm, desc='Гармонический сигнал')
    plt.subplot(gs[1, :])
    plot(noise, desc='Шумы')
    plt.subplot(gs[2, :])
    plot(noise + harm, desc='Гармонический сигнал с шумами')
    plt.show()

    gs = gr.GridSpec(6, 6)
    realisation = np.zeros(1000)
    j = 0
    for i in m:
        # plt.subplot(gs[j, :])
        realisation += normalize(numpy_random(n), s) + harm
        realisation /= i
        plot(realisation, desc='M = ' + str(i))
        j += 1
        print('M = ' + str(i) + ', std: ' + str(np.std(realisation)))
    plt.show()


#   120 - фаза сжимания сердца
def ten_task(x, m):
    gs = gr.GridSpec(3, 2)
    heart = heartbeat()
    card = cardiography(m)
    conv = convolution(heart, card)
    ft_h = fourier_transform(heart, len(heart))
    ft_c = fourier_transform(card, len(card))
    ft_con = fourier_transform(conv, len(conv))
#   тики раз в секунду: 110-130
    plt.subplot(gs[0, 0])
    plot(heart, desc='Биение сердца')
    plt.subplot(gs[0, 1])
    plot(ft_h[1][:len(ft_h[1]) // 2], desc='Fourier for heartbeat')
    plt.subplot(gs[1, 0])
    plot(card, desc='Кардиограмма')
    plt.subplot(gs[1, 1])
    plot(ft_c[1][:len(ft_c[1]) // 2], desc='Fourier for cardiograpthy')
    plt.subplot(gs[2, 0])
    plot(conv, desc='Свертка')
    plt.subplot(gs[2, 1])
    plot(ft_con[1][:len(ft_con[1]) // 2], desc='Fourier for convolution') #cs vs cn ??
    plt.show()


def eleven_task(dt=0.002, m=128, fcut=16.32):
    gs = gr.GridSpec(2, 3)
    _sum_harm = harmonic_motion(1000, a=10, f=5, t=0.001) + harmonic_motion(1000, a=10, f=50, t=0.001) + \
                harmonic_motion(1000, a=10, f=150, t=0.001)
    lpF = low_pass_filter(fcut=fcut)
    conv = convolution(_sum_harm, lpF)
    ft_sh = fourier_transform(_sum_harm, len(_sum_harm))
    ft_con = fourier_transform(conv, len(conv))
    ft_lpF = fourier_transform(lpF, len(lpF))
    plt.subplot(gs[0, 0])
    plot(_sum_harm, desc='Сумма гармоник')
    plt.subplot(gs[0, 1])
    plot(conv, desc='Свертка от суммы и фильтра')
    plt.subplot(gs[0, 2])
    plot(lpF, desc='ФНЧ')
    plt.subplot(gs[1, 0])
    plot(ft_sh[1], desc='Фурье от суммы гармоник')
    plt.subplot(gs[1, 1])
    plot(ft_con[1], desc='Фурье от свертки')
    plt.subplot(gs[1, 2])
    plot(ft_lpF[1] * 2 * m, desc='Частотная характеристика')
    plt.show()


def twelve_task(dt=0.002, m=128, fcut=15):
    gs = gr.GridSpec(2, 4) #(2, 4)
    _sum_harm = harmonic_motion(1000, a=10, f=5, t=0.001) + harmonic_motion(1000, a=10, f=50, t=0.001) + \
                harmonic_motion(1000, a=10, f=150, t=0.001)
    lpF = low_pass_filter(fcut=fcut)
    hpF = high_pass_filter(fcut=fcut)
    bpF = band_pass_filter(10, 30)
    bsF = band_stop_filter(10, 30)
    lpF_conv = convolution(_sum_harm, lpF)
    hpF_conv = convolution(lpF_conv, hpF)
    bpF_conv = convolution(hpF_conv, bpF)
    bsF_conv = convolution(lpF_conv, bsF)
    ft_lpF_harm = fourier_transform(_sum_harm, len(_sum_harm))
    ft_hpF_harm = fourier_transform(hpF_conv, len(hpF_conv))
    ft_bpF_harm = fourier_transform(bpF_conv, len(bpF_conv))
    ft_bsF_harm = fourier_transform(bsF_conv, len(bsF_conv))
    ft_lpF = fourier_transform(lpF, len(lpF))
    ft_hpF = fourier_transform(hpF, len(hpF))
    ft_bpF = fourier_transform(bpF, len(bpF))
    ft_bsF = fourier_transform(bsF, len(bsF))
    plt.subplot(gs[0, 0])
    plot(lpF, desc='ФНЧ')
    plt.subplot(gs[0, 1])
    plot(hpF, desc='ФВЧ')
    plt.subplot(gs[0, 2])
    plot(bpF, desc='ППЧ')
    plt.subplot(gs[0, 3])
    plot(bsF, desc='РФ')
    plt.subplot(gs[1, 0])
    #исправить вывод
    plot(ft_lpF[1] * 2 * m, desc='Частотная хар-ка ФНЧ')
    plt.subplot(gs[1, 1])
    plot(ft_hpF[1] * 2 * m, desc='Частотная хар-ка ФВЧ')
    plt.subplot(gs[1, 2])
    plot(ft_bpF[1] * 2 * m, desc='Частотная хар-ка ППЧ')
    plt.subplot(gs[1, 3])
    plot(ft_bsF[1], desc='Частотная хар-ка РФ')
    plt.show()
    plt.subplot(gs[0, 0])
    plot(lpF_conv[:len(lpF_conv) // 2], desc='ФНФ')
    plt.subplot(gs[0, 1])
    plot(hpF_conv[:len(hpF_conv) // 2], desc='ФВФ')
    plt.subplot(gs[0, 2])
    plot(bpF_conv[:len(bpF_conv) // 2], desc='ППФ')
    plt.subplot(gs[0, 3])
    plot(bsF_conv[:len(bsF_conv) // 2], desc='РФ')
    plt.subplot(gs[1, 0])
    plot(ft_lpF_harm[1] * 2 * m, desc='Частотная хар-ка ФНФ')
    plt.subplot(gs[1, 1])
    plot(ft_hpF_harm[1] * 2 * m, desc='Частотная хар-ка ФВФ')
    plt.subplot(gs[1, 2])
    plot(ft_bsF_harm[1], desc='Частотная хар-ка РФ')
    plt.subplot(gs[1, 3])
    plot(ft_bpF_harm[1] * 2 * m, desc='Частотная хар-ка ППФ')
    plt.show()


def thirteen_task():
    gs = gr.GridSpec(2, 1)
    wav = wav_values()
    plt.subplot(gs[0, :])
    plot(wav, desc='Запись')
    plt.subplot(gs[1, :])
    ft_wav = fourier_transform(wav, len(wav))
    plot(ft_wav, desc='Фурье записи')
    plt.show()


def course_work():
    gs = gr.GridSpec(1, 2)
    mod = modulation()
    noise = numpy_random(1000)
    plt.subplot(gs[0, 0])
    plot(normalize(signal(), 2), desc='Исходный сигнал')
    plt.subplot(gs[0, 1])
    plot(normalize(carrying_oscillation(), 2), desc='Несущая')
    plt.show()
    gs = gr.GridSpec(2, 2)
    plt.subplot(gs[0, 0])
    plot(normalize(mod, 2), desc='Модуляция при m=0.5')
    plt.subplot(gs[0, 1])
    plot(normalize(modulation(m=0.1), 2), desc='модуляция при m=0.1')
    plt.subplot(gs[1, 0])
    plot(normalize(modulation(m=0.99), 2), desc='модуляция при m=0.99')
    plt.subplot(gs[1, 1])
    plot(normalize(modulation(m=1.5), 2), desc='модуляция при m=1.5')
    plt.show()
    gs = gr.GridSpec(1, 2)
    plt.subplot(gs[0, 0])
    plot(normalize(mod, 2), desc='Модуляция при m=0.5')
    plt.subplot(gs[0, 1])
    plot(normalize(mod + noise, 2), desc='Сигнал с шумами')
    plt.show()
    ft_mod_sig = fourier_transform(mod, len(mod))
    ft_mod_noise_sig = fourier_transform(mod + noise, len(mod))
    plt.subplot(gs[0, 0])
    plot(ft_mod_sig[1], desc='Спектр')
    plt.subplot(gs[0, 1])
    plot(ft_mod_noise_sig[1], desc='Спектр с шумами')
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
    # nine_task(N)
    # ten_task(x, 200)
    # eleven_task(fcut=30)
    # twelve_task(fcut=30)
    # thirteen_task()
    course_work()

