import numpy as np

def calculate_camera_matrix(K, R, C):
    """
    Calculate the camera matrix P = KR[I | -C]
    
    Args:
    K (np.array): 3x3 camera intrinsic matrix
    R (np.array): 3x3 rotation matrix
    C (np.array): 3x1 camera center

    Returns:
    np.array: 3x4 camera matrix P
    """
    # Create the identity matrix
    I = np.eye(3)
    
    # Create [I | -C]
    IC = np.hstack((I, -C))
    
    # Calculate KR[I | -C]
    P = K @ R @ IC
    
    return P

# Define the matrices
K = np.array([
    [7.03606595e+03, 1.55093658e+02, 4.04916599e+03],
    [0, 7.11020543e+03, 9.01944958e+01],
    [0, 0, 1]
])

R = np.array([
    [0.24363231, -0.966366, 0.08234101],
    [-0.07937672, -0.10448198, -0.99135405],
    [0.966614, 0.23498992, -0.10216215]
])

C = np.array([
    [-739.8902994],
    [-485.37784529],
    [321.63665888]
])

# Calculate camera matrix
P = calculate_camera_matrix(K, R, C)

# Print results
print("Camera Matrix P = KR[I | -C]:")
print(P)

# Print the [I | -C] matrix for verification
I = np.eye(3)
IC = np.hstack((I, -C))
print("\n[I | -C] matrix:")
print(IC)
