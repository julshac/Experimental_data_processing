from subprocess import Popen
import numpy as np
from scipy import misc
from scipy.misc import imsave
from scipy.io.wavfile import read


def dat_values():
    # p = Popen("pgp_1ms.dat", cwd=r"C:\OSPanel\Works")
    # stdout, stderr = p.communicate()
    a = np.fromfile("data/pgp_1ms.dat", dtype=np.float32)
    print(a)
    return a


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
