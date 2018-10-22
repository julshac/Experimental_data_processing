import matplotlib.pyplot as plt
import seaborn as sns


def plot(x, y=None, desc="", xlabel="x", ylabel="f(x)"):
    if y is None:
        plt.plot(x)
    else:
        plt.plot(x, y)
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
    sns.distplot(x, bins=100, norm_hist=0, kde=0)
    plt.title(desc)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()
