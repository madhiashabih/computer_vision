# import cv2
# import matplotlib.pyplot as plt 
# import numpy as np
# from typing import List, Tuple
# plt.rcParams['figure.figsize'] = [15, 15]
# from mpl_toolkits.mplot3d import Axes3D
# import random
# from q1 import plot_matches, ransac, triangulate_point, draw_plot_3d, check_chirality, filter_matches, decomposeP
# from q2 import calculate_r1, calculate_r2, calculate_r3, calculate_T1, calculate_T2
# from applyhomography import applyhomography
# from PIL import Image
# import numpy.linalg as la

# def main():
#     ########## QUESTION 1 ###########
#     ########## Q1a ########## 
#     # Load data
#     data = np.loadtxt('ET/matches.txt')
#     src_pts, dst_pts = data[:, :2], data[:, 2:]
    
#     src_img = cv2.imread('ET/et1.jpg', cv2.IMREAD_COLOR)
#     dst_img = cv2.imread('ET/et1.jpg', cv2.IMREAD_COLOR)
#     matches = np.hstack((src_pts, dst_pts))
    
#     # Plot matches
#     matches = filter_matches(matches, max_distance= 150)
#     plot_matches(src_img, matches, 150)
    
#     ########## Q1b ##########
#     # RANSAC
#     inliers_1, inliers_2, F = ransac(matches, threshold=1, iters=2000)
#     inliers = np.hstack((inliers_1, inliers_2))
#     plot_matches(src_img, inliers, max_distance=150)
#     print("\nFundamental matrix F:")
#     print(F)
    
#     ########## Q1c ##########
#     # Load camera matrix K
#     K = np.loadtxt('ET/K.txt')
#     print("\nCamera matrix K:")
#     print(K)
    
#     # Calculate Essential matrix E
#     E = K.T @ F @ K
    
#     # SVD of E
#     U, S, Vt = np.linalg.svd(E)
    
#     det_U = np.linalg.det(U)
#     det_Vt = np.linalg.det(Vt)

#     if det_U > 0 and det_Vt < 0:
#         E = -E
#         Vt = -Vt
#     if det_U < 0 and det_Vt > 0:
#         E = -E
#         U = -U
    
#     print("\nEssential matrix E:")
#     print(E)
#     print("\nMatrix U:")
#     print(U)
#     print("\nSingular values (S):")
#     print(S)
#     print("\nMatrix V^T:")
#     print(Vt)
    
#     ########## Q1d ##########  
#     # Calculate projection matrices
#     R = np.eye(3)
#     P = K @ np.hstack([R, np.zeros((3, 1))])
#     print("\nProjection matrix P:")
#     print(P)
    
#     W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
#     u3 = U[:, 2].reshape(-1, 1)
    
#     potential_P_primes = [
#         K @ np.hstack([U @ W @ Vt, u3]),
#         K @ np.hstack([U @ W @ Vt, -u3]),
#         K @ np.hstack([U @ W.T @ Vt, u3]),
#         K @ np.hstack([U @ W.T @ Vt, -u3])
#     ]
    
#     # Check chirality condition
#     n1 = np.eye(3).T @ np.array([0, 0, 1]).reshape(-1, 1) # result: 3x1 array
#     C1 = np.zeros((3, 1))
    
#     P_prime = None
#     R_prime = None
#     for P_possible_prime in potential_P_primes:
#         X = triangulate_point(P, P_possible_prime, inliers_1[0], inliers_2[0])
        
#         R_possible_prime = P_possible_prime[:,:3]
#         C2 = -np.linalg.inv(R_possible_prime) @ P_possible_prime[:, 3]
#         n2 = R_possible_prime.T @ np.array([0, 0, 1])
        
#         print(f"X: {X}")
#         print(f"n1: {n1}")
#         print(f"C1: {C1}")
#         print(f"n2: {n2}")
#         print(f"C2: {C2}")
        
#         if check_chirality(X.reshape(-1,1), n1, C1, n2.reshape(-1,1), C2.reshape(-1,1)):
#             P_prime = P_possible_prime
#             R_prime = R_possible_prime 
#     print("\nP-prime")
#     print(P_prime)        
     
#     ########## Q1e ##########    
        
#     # inliers_1 = np.array(inliers_1)
#     # inliers_2 = np.array(inliers_2)
#     # draw_plot_3d(inliers_1, inliers_2, P, P_prime, src_img, dst_img, "output/et_3d")

#     ########## QUESTION 2 ###########
    
#     K1, R1, c1 = decomposeP(P)
#     K2, R2, c2 = decomposeP(P_prime)  
      
#     Kn = 0.5 * (K1 + K2)
#     print(f"Kn: {Kn}")
#     # k 
#     k = (R1.T @ np.array([0, 0, 1]))

#     print(f"C1: {C1}")
#     print(f"C2: {C2}")
#     print(f"k: {k}")
    
#     # # Calculate r's
#     # r1 = calculate_r1(C1, C2.reshape(-1,1))
#     # r2 = calculate_r2(k, r1)
#     # r3 = calculate_r3(r1, r2)

#     # print(f"r1: {r1}")
#     # print(f"r2: {r2}")
#     # print(f"r3: {r3}")
    
#     # # Construct Rn
#     # Rn = np.vstack((r1, r2, r3))

#     # print(f"Rn: {Rn}")
    
#     # # Construct T1 and T2
#     # T1 = calculate_T1(Kn, Rn, R1, K1)
#     # T2 = calculate_T2(Kn, Rn, R2, K2)

#     # print(f"T1: {T1}")
#     # print(f"T2: {T2}")
    
#     r1 = (c2 - c1) / la.norm(c2 - c1)
#     r2 = np.cross(k, r1) / la.norm(np.cross(k, r1))
#     r3 = np.cross(r1, r2)
#     Rn = np.vstack((r1, r2, r3))

#     T1 = Kn @ Rn @ R1.T @ la.inv(K1)
#     T2 = Kn @ Rn @ R2.T @ la.inv(K2)

#     # print(f"T1: {T1}")
#     # print(f"T2: {T2}")
    
#     x1, y1, transformed_1 = applyhomography(src_img, T1)
#     x2, y2, transformed_2 = applyhomography(dst_img, T2)
    
#     cv2.imwrite("output/et1_t.jpg", transformed_1)
#     cv2.imwrite("output/et2_t.jpg", transformed_2)
    
# if __name__ == "__main__":
#     main()


import matplotlib.pyplot as plt 
import numpy as np


# Import functions from other modules (assuming they exist)
from functions import load_data, process_matches, compute_fundamental_matrix, compute_essential_matrix, compute_projection_matrices, find_valid_projection_matrix, compute_rectification_transforms, apply_rectification, save_images
from applyhomography import applyhomography

plt.rcParams['figure.figsize'] = [15, 15]

def main():
    # Load data
    matches, src_img, dst_img = load_data('ET/matches.txt', 'ET/et1.jpg')
    
    # Process matches
    filtered_matches = process_matches(matches, src_img)
    
    # Compute fundamental matrix
    inliers_1, inliers_2, F = compute_fundamental_matrix(src_img, filtered_matches)
    print("\nFundamental matrix F:")
    print(F)
    
    # Load camera matrix K
    K = np.loadtxt('ET/K.txt')
    print("\nCamera matrix K:")
    print(K)
    
    # Compute essential matrix
    E, U, S, Vt = compute_essential_matrix(F, K)
    print("\nEssential matrix E:")
    print(E)
    
    print("\nMatrix U:")
    print(U)
    print("\nSingular values (S):")
    print(S)
    print("\nMatrix V^T:")
    print(Vt)
    
    # Compute projection matrices
    P, potential_P_primes = compute_projection_matrices(K, U, Vt)
    print("\nProjection matrix P:")
    print(P)
    
    # Find valid projection matrix
    P_prime, R_prime = find_valid_projection_matrix(P, potential_P_primes, inliers_1, inliers_2)
    print("\nP-prime:")
    print(P_prime)
    
    # Compute rectification transforms
    T1, T2 = compute_rectification_transforms(P, P_prime)
    
    # Apply rectification
    transformed_1, transformed_2 = apply_rectification(src_img, dst_img, T1, T2)
    
    # Save transformed images
    save_images(transformed_1, transformed_2, "output/et1_t.jpg", "output/et2_t.jpg")

if __name__ == "__main__":
    main()