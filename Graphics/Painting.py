import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def vector(x, desc, xlabel="x", ylabel="f(x)"):
    plt.plot(x)
    plt.grid()
    plt.title(desc)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def histogram(x, desc):
    plt.hist(x)
    plt.title(desc)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()


def distribution(x, desc):
    sns.set(color_codes=True)
    sns.distplot(x)
    plt.title(desc)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()
