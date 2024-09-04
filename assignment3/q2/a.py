import numpy as np

# Define the vectors
C1 = np.array([-739.8902994, -485.37784529, 321.63665888])
C2 = np.array([-721.05444208, -1082.93989766, 380.46969249])

# Calculate the Euclidean distance
distance = np.linalg.norm(C2 - C1)

print("Euclidean distance:", distance)

import numpy as np

# Define the rotation matrices
R1 = np.array([
    [0.24363231, -0.966366, 0.08234101],
    [-0.07937672, -0.10448198, -0.99135405],
    [0.966614, 0.23498992, -0.10216215]
])

R2 = np.array([
    [-0.82783123, 0.5608882, -0.00999388],
    [0.05685095, 0.10160475, 0.9931991],
    [-0.55808908, -0.82163307, 0.11599862]
])

# Extract the first two columns
col1_1 = R1[:, 0]
col2_1 = R1[:, 1]
col1_2 = R2[:, 0]
col2_2 = R2[:, 1]

# Compute the dot product and magnitudes
def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    # Clip the value to avoid numerical issues with arccos
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    return np.degrees(angle)

# Calculate the angles
angle_col1 = angle_between_vectors(col1_1, col1_2)
angle_col2 = angle_between_vectors(col2_1, col2_2)

print(f"Angle between first columns: {angle_col1:.2f} degrees")
print(f"Angle between second columns: {angle_col2:.2f} degrees")

