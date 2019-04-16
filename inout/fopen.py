import struct
from subprocess import Popen
import numpy as np
from scipy import misc
from scipy.misc import imsave
from scipy.io.wavfile import read
import io


def dat_values(path, length=76):
    # noise = []
    # with open(path, 'rb') as f:
    #     for i in range(length):
    #         d = f.read(4)
    #         t = struct.unpack('f', d)
    #         noise.append(t)
    with open(path, 'br') as f:
        data = np.array(struct.unpack(str(length) + "f", f.read()), dtype=np.float32)
    return data


def dat_2D_reader(path, rows=221, cols=307):
    with open(path, 'br') as f:
        data = np.array(struct.unpack(str(rows * cols) + "f", f.read()), dtype=np.float16)
        data = data.reshape((rows, cols))
    return data


def wav_values():
    a = read("data/Recording.wav")
    return np.array(a[1], dtype=float)


def img_values(path):
    img = misc.imread(path)
    return img


def to_one_channel(picture):
    if len(picture.shape) > 2:
        return picture[:, :, 0]
    else:
        return picture


def xcr_values(path):
    xcr = np.fromfile(path, dtype=np.uint16)
    return xcr.reshape((300, 400))
