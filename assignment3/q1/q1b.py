import numpy as np

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


# Create the identity matrix
I = np.eye(3)

join = np.hstack((I, -C))

# Calculate KR[I - C]
result = K @ R @ (join)

print(join)

#scaling_factor = result[2,2]
#result_scaled = result/ scaling_factor

print("KR[I | - C] =")
print(result)
