import numpy as np
import numpy.linalg as la

def calculate_r1(C1, C2):
    diff = C2 - C1
    return diff / np.linalg.norm(diff)

def calculate_r2(k, r1):
    print(f"k: {k}")
    print(f"r1: {r1}")
    kr1 = np.cross(k, r1)
    return kr1 / np.linalg.norm(kr1)

def calculate_r3(r1, r2):
    return np.cross(r1, r2)


def calculate_T1(Kn, Rn, R1, K1):
    return Kn @ Rn @ R1.T @ la.inv(K1)

def calculate_T2(Kn, Rn, R2, K2):
    return Kn @ Rn @ R2.T @ la.inv(K2)

