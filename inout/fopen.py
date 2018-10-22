from subprocess import Popen
import numpy as np


def dat_values():
    # p = Popen("pgp_1ms.dat", cwd=r"C:\OSPanel\Works")
    # stdout, stderr = p.communicate()
    return np.fromfile("C:\\Users\\Julia\\PycharmProjects\\MetProcesExpData_1\\inout\\pgp_1ms.dat", dtype=float)
