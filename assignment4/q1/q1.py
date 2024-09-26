import cv2
import matplotlib.pyplot as plt 
import numpy as np
from typing import List, Tuple
plt.rcParams['figure.figsize'] = [15, 15]
from mpl_toolkits.mplot3d import Axes3D
import random
from functions import plot_matches, ransac, decomposeP, triangulate_point, check_chirality, triangulate_point_array, plot_3d_points


def main():
    ########## Q1a ########## 
    # Load data
    data = np.loadtxt('ET/matches.txt')
    src_pts, dst_pts = data[:, :2], data[:, 2:]
    
    src_img = cv2.imread('ET/et1.jpg', cv2.IMREAD_COLOR)
    matches = np.hstack((src_pts, dst_pts))
    
    # Plot matches
    filtered_matches = plot_matches(src_img, matches, max_distance=150)
    print("\nFiltered matches:")
    print(filtered_matches)
    
    ########## Q1b ##########
    # RANSAC
    inliers, F = ransac(filtered_matches, threshold=1, iters=2000)
    print("\n inliers:")
    print(inliers)
    plot_matches(src_img, inliers, max_distance=1000)
    print("\nFundamental matrix F:")
    print(F)
    
    ########## Q1c ##########
    # Load camera matrix K
    K = np.loadtxt('ET/K.txt')
    print("\nCamera matrix K:")
    print(K)
    
    # Calculate Essential matrix E
    E = K.T @ F @ K
    print("\nEssential matrix E:")
    print(E)
    
    # SVD of E
    U, S, Vt = np.linalg.svd(E)
    
    if np.linalg.det(U) * np.linalg.det(Vt.T) < 0:
        E = -E
        U, S, Vt = np.linalg.svd(E)
    
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
    
    W = np.array([[1, -1, 0], [1, 0, 0], [0, 0, 1]])
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
    
    for i, P_prime in enumerate(potential_P_primes):
        _, _, C2 = decomposeP(P_prime)
        n2 = P_prime[:, :3] @ np.array([0, 0, 1]).reshape(-1, 1)
        
        X = triangulate_point(P, P_prime, src_pts[0], dst_pts[0])
        
        if check_chirality(X.reshape(-1, 1), n1, C1, n2, C2.reshape(-1, 1)):
            print(f"\nValid P' found (option {i+1}):")
            print(P_prime)
             
            # Plot 3D graph 
            X = triangulate_point_array(P, P_prime, src_pts, dst_pts)
            
            break
    else:
        print("No valid P' found satisfying the chirality condition.")
     
    ########## Q1e ##########    
        
    plot_3d_points(X, P, P_prime)

if __name__ == "__main__":
    main()