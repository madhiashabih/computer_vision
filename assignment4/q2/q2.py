import numpy as np

def calculate_r1(C1, C2):
    diff = C2 - C1
    return diff / np.linalg.norm(diff)

def calculate_r2(k, r1):
    kr1 = np.cross(k, r1)
    return kr1 / np.linalg.norm(kr1)

def calculate_r3(r1, r2):
    return np.cross(r1, r2)


def calculate_T1(Kn, Rn, R1, K1_inv):
    return np.dot(np.dot(np.dot(Kn, Rn), R1.T), K1_inv)

def calculate_T2(Kn, Rn, R2, K2_inv):
    return np.dot(np.dot(np.dot(Kn, Rn), R2.T), K2_inv)

# Camera matrices:
P1 = []
P2 = []

# Rotation matrices:
R1 = []
R2 = []

# Calibration matrices:
K1 = []
K2 = []

Kn = 0.5 * (K1 + K2)

# k 
k = [0][0][1]

# Calculate r's
r1 = calculate_r1(C1, C2)
r2 = calculate_r2(k, r1)
r3 = calculate_r3(r1, r2)

# Construct Rn
Rn = np.vstack((r1, r2, r3))

# Construct T1 and T2
T1 =(Kn, Rn, R1, K1)
T2 =(Kn, Rn, R2, K2)
