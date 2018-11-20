from subprocess import Popen
import numpy as np
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
