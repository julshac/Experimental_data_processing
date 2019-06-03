import matplotlib.pyplot as plt
# import seaborn as sns


def plot(x, y=None, desc="", xlabel="x", ylabel="f(x)"):
    if y == None:
        plt.plot(x)
    else:
        plt.plot(x, y)
    plt.grid()
    plt.title(desc)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.show()


def histogram(x, desc):
    plt.hist(x)
    plt.title(desc)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    # plt.show()


def distribution(x, desc):
    sns.distplot(x, bins=100, norm_hist=0, kde=0)
    plt.title(desc)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    # plt.show()


# def plot_many(x, y, desc="", xlabel="x", ylabel="f(x)"):
#     # data = np.genfromtxt('csv_file', delimiter=',', dtype=float)
#     columns = x.shape[1] - 1 # a numpy array 'shape' is (rows, columns, etc). I am subtracting the x column.
#     nrows, ncols = 4, 2 # the product of these should be number of plots (I'm assuming columns=8)
#
#     fig = plt.figure()
#     x = data[:, 0] # same as [row[0] for row in data], and doesn't change in loop, so only do it once.
#     for m in range(1, columns +1 ): # this loops from 1 through columns
#         y = data[:, m] # same as [row[m] for row in data]
#         ax = fig.add_subplot(nrows, ncols, m, axisbg='w')
#         ax.plot(x, y, lw=1.3)
#
#     plt.show() # after loop finishes, only need to show once.
