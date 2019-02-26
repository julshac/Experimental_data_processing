import matplotlib.gridspec as gr
from inout.plot import *
from inout.fopen import *
from analysis.image_analysis import *
from analysis.amplitude_modulation import *


def min_image_statistics_output(picture):
    # дисперсию, рседнее значение, минимум, максимум
    min = picture.min()
    max = picture.max()
    std = picture.std()
    print(f"Min: {min}, Max: {max}, Std: {std}")
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
    print(normalize(picture, 1))
    #гистограмма яркости
    plt.hist(picture.flatten(), 255)


def result():
    picture = img_values()
    plt.figure()
    plt.imshow(picture)
    plt.figure()
    min_image_statistics_output(picture)
    brightness_output(picture)







