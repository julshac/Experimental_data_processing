from subprocess import Popen
import numpy as np


def dat_values():
    # p = Popen("pgp_1ms.dat", cwd=r"C:\OSPanel\Works")
    # stdout, stderr = p.communicate()
    return np.fromfile("data/pgp_1ms.dat", dtype=float)
