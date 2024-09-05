import numpy as np

# Define the vectors
C1 = np.array([-739.8902994, -485.37784529, 321.63665888])
C2 = np.array([-721.05444208, -1082.93989766, 380.46969249])

# Calculate the Euclidean distance
distance = np.linalg.norm(C2 - C1)

print("Euclidean distance:", distance)

def calculate_angle(R1, R2):
    # Calculate the dot product of R1 and R2
    dot_product = np.trace(np.dot(R1, R2.T))
    
    # Calculate the magnitudes of R1 and R2
    magnitude_R1 = np.sqrt(np.trace(np.dot(R1, R1.T)))
    magnitude_R2 = np.sqrt(np.trace(np.dot(R2, R2.T)))
    
    # Calculate cos(theta)
    cos_theta = dot_product / (magnitude_R1 * magnitude_R2)
    
    # Calculate theta in radians
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    
    # Convert theta to degrees
    theta_deg = np.degrees(theta_rad)
    
    return theta_deg

# Define the matrices
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

# Calculate the angle
angle = calculate_angle(R1[:1], R2[:1])

print(f"The angle between R1 and R2 is approximately {angle:.4f} degrees.")



