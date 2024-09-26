import cv2
import matplotlib.pyplot as plt 
import numpy as np
from typing import List, Tuple
plt.rcParams['figure.figsize'] = [15, 15]
from mpl_toolkits.mplot3d import Axes3D
import random
from functions import plot_matches, ransac, decomposeP, triangulate_point, check_chirality, triangulate_point_array, plot_3d_points, is_point_in_front, filter_matches


def main():
    ########## Q1a ########## 
    # Load data
    data = np.loadtxt('ET/matches.txt')
    src_pts, dst_pts = data[:, :2], data[:, 2:]
    
    src_img = cv2.imread('ET/et1.jpg', cv2.IMREAD_COLOR)
    matches = np.hstack((src_pts, dst_pts))
    
    # Plot matches
    matches = filter_matches(matches, max_distance)
    plot_matches(src_img, matches, max_distance=150)
    
    ########## Q1b ##########
    # RANSAC
    inliers_1, inliers_2, F = ransac(filtered_matches, threshold=1, iters=2000)
    inliers = np.hstack((inliers_1, inliers_2))
    plot_matches(src_img, inliers, max_distance=150)
    print("\nFundamental matrix F:")
    print(F)
    
    ########## Q1c ##########
    # Load camera matrix K
    K = np.loadtxt('ET/K.txt')
    print("\nCamera matrix K:")
    print(K)
    
    # Calculate Essential matrix E
    E = K.T @ F @ K
    
    # SVD of E
    U, S, Vt = np.linalg.svd(E)
    
    det_U = np.linalg.det(U)
    det_Vt = np.linalg.det(Vt)

    if det_U > 0 and det_Vt < 0:
        E = -E
        Vt = -Vt
    if det_U < 0 and det_Vt > 0:
        E = -E
        U = -U
    
    print("\nEssential matrix E:")
    print(E)
    print("\nMatrix U:")
    print(U)
    print("\nSingular values (S):")
    print(S)
    print("\nMatrix V^T:")
    print(Vt)
    
    ########## Q1d ##########  
    # Calculate projection matrices
    P = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    print("\nProjection matrix P:")
    print(P)
    
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    u3 = U[:, 2].reshape(-1, 1)
    
    potential_P_primes = [
        K @ np.hstack([U @ W @ Vt, u3]),
        K @ np.hstack([U @ W @ Vt, -u3]),
        K @ np.hstack([U @ W.T @ Vt, u3]),
        K @ np.hstack([U @ W.T @ Vt, -u3])
    ]
    
    # Check chirality condition
    n1 = np.eye(3).T @ np.array([0, 0, 1]).reshape(-1, 1) # result: 3x1 array
    C1 = np.zeros((3, 1))
    
    P_prime = none
    for P_possible_prime in potential_P_primes:
        X = triangulate_point(P, P_possible_prime, inliers_1[0], inliers_2[0])
        cam1 = is_point_in_front(P, inliers_1)
        cam2 = is_point_in_front(P_possible_prime, inliers_2)
        
        if cam1 and cam2:
            P_prime = P_possible_prime
            
    print("\nP-prime")
    print(P_prime)       
     
    ########## Q1e ##########    
        
    plot_3d_points(X, P, P_possible_prime)

if __name__ == "__main__":
    main()