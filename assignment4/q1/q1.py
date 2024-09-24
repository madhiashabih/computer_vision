# import cv2
# import matplotlib.pyplot as plt 
# import numpy as np
# import random
# plt.rcParams['figure.figsize'] = [15, 15]
# from applyhomography import applyhomography
# from PIL import Image  

# def calculate_distance(x1, y1, x2, y2):
#     return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# def filter_matches(matches, max_distance):
#     return [match for match in matches if calculate_distance(*match) <= max_distance]

# def random_point(matches, k=4):
#     idx = random.sample(range(len(matches)), k)
#     point = [matches[i] for i in idx ]
#     return np.array(point)

# def get_error(points, H):
#     num_points = len(points)
#     all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
#     all_p2 = points[:, 2:4]
#     estimate_p2 = np.zeros((num_points, 2))
#     for i in range(num_points):
#         temp = np.dot(H, all_p1[i])
#         estimate_p2[i] = (temp/temp[2])[0:2] # set index 2 to 1 and slice the index 0, 1
#     # Compute error
#     errors = np.linalg.norm(all_p2 - estimate_p2 , axis=1) ** 2
#     return errors

# def homography(pairs):
#     rows = []
#     for i in range(pairs.shape[0]):
#         p1 = np.append(pairs[i][0:2], 1)
#         p2 = np.append(pairs[i][2:4], 1)
#         row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
#         row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
#         rows.append(row1)
#         rows.append(row2)
#     rows = np.array(rows)
#     U, s, V = np.linalg.svd(rows)
#     H = V[-1].reshape(3, 3)
#     H = H/H[2, 2] # standardize to let w*H[2,2] = 1
#     return H

# def plot_matches(src_img, matches, max_distance):
#     # Convert the source image to grayscale
#     gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    
#     filtered_matches = filter_matches(matches, max_distance)
#     # Create a figure and axis
#     fig, ax = plt.subplots(figsize=(15, 15))
    
#     # Display the grayscale image
#     ax.imshow(gray_img, cmap='gray')
    
#     # Plot the matches
#     for match in filtered_matches:
#         x1, y1, x2, y2 = match
#         ax.plot([x1, x2], [y1, y2], 'r-', linewidth=0.5)
#         ax.plot(x1, y1, 'bo', markersize=5)  # Blue circle for xi
    
#     ax.set_title(f'Feature Matches on Grayscale Image (Max Distance: {max_distance:.2f})')
#     ax.axis('off')
#     plt.tight_layout()
#     plt.show()

# def ransac(matches, threshold, iters):
#     num_best_inliers = 0
    
#     for i in range(iters):
#         points = random_point(matches)
#         H = homography(points)
        
#         #  avoid dividing by zero 
#         if np.linalg.matrix_rank(H) < 3:
#             continue
            
#         errors = get_error(matches, H)
#         idx = np.where(errors < threshold)[0]
#         inliers = matches[idx]

#         num_inliers = len(inliers)
#         if num_inliers > num_best_inliers:
#             best_inliers = inliers.copy()
#             num_best_inliers = num_inliers
#             #best_H = H.copy()
#             best_H = homography(best_inliers)

#     print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
#     return best_inliers, best_H

# def find_homography(pts_src: np.ndarray, pts_dst: np.ndarray) -> np.ndarray:
#     """Calculate the homography matrix using SVD."""
#     n = pts_src.shape[0]
#     A = np.zeros((2*n, 12))
    
#     for i in range(n):
#         x, y = pts_src[i]
#         X, Y, Z = pts_dst[i]
#         A[2*i] = [0, 0, 0, 0, -X, -Y, -Z, -1, y*X, y*Y, y*Z, y]
#         A[2*i+1] = [X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x]
    
#     _, _, Vh = np.linalg.svd(A)
#     return Vh[-1].reshape(3, 4)

# def decomposeP(P):
#     '''
#         The input P is assumed to be a 3-by-4 homogeneous camera matrix.
#         The function returns a homogeneous 3-by-3 calibration matrix K,
#         a 3-by-3 rotation matrix R and a 3-by-1 vector c such that
#         K*R*[eye(3), -c] = P.

#     '''

#     W = np.array([[0, 0, 1],
#                   [0, 1, 0],
#                   [1, 0, 0]])

#     # calculate K and R up to sign
#     Qt, Rt = np.linalg.qr((W.dot(P[:,0:3])).T)
#     K = W.dot(Rt.T.dot(W))
#     R = W.dot(Qt.T)

#     # correct for negative focal length(s) if necessary
#     D = np.array([[1, 0, 0],
#                   [0, 1, 0],
#                   [0, 0, 1]])
#     if K[0,0] < 0:
#         D[0,0] = -1
#     if K[1,1] < 0:
#         D[1,1] = -1
#     if K[2,2] < 0:
#         D[2,2] = -1
#     K = K.dot(D)
#     R = D.dot(R)

#     # calculate c
#     c = -R.T.dot(np.linalg.inv(K).dot(P[:,3]))

#     return K, R, c

# def triangulate_point(P1, P2, x1, x2):
#     """
#     Triangulate a 3D point from two 2D correspondences and projection matrices.
    
#     Args:
#     - P1: 3x4 projection matrix of the first camera
#     - P2: 3x4 projection matrix of the second camera
#     - x1: 2D point in the first image (shape: (2,))
#     - x2: 2D point in the second image (shape: (2,))
    
#     Returns:
#     - X: Triangulated 3D point (homogeneous coordinates)
#     """
    
#     # Build the system of equations (Ax = 0)
#     A = np.zeros((4, 4))
#     A[0] = x1[0] * P1[2] - P1[0]  # x1[0] * P1's 3rd row - P1's 1st row
#     A[1] = x1[1] * P1[2] - P1[1]  # x1[1] * P1's 3rd row - P1's 2nd row
#     A[2] = x2[0] * P2[2] - P2[0]  # x2[0] * P2's 3rd row - P2's 1st row
#     A[3] = x2[1] * P2[2] - P2[1]  # x2[1] * P2's 3rd row - P2's 2nd row

#     # Solve using SVD (Ax = 0 -> Use SVD to find the solution for X)
#     _, _, V = np.linalg.svd(A)
#     X = V[-1]  # The solution is the last row of V

#     # Convert X from homogeneous coordinates to 3D by dividing by X[3]
#     X = X / X[3]
    
#     return X[:3]

# def check_chirality(X, n1, C1, n2, C2):
#     print("\nX.T:")
#     print(X.reshape(-1,1))
#     print("\nC1")
#     print(C1)
#     print("\nn1.T:")
#     print(n1)

#     test1 = np.dot(n1.T, (X.reshape(-1,1)-C1)) > 0
#     test2 = np.dot(n2.T, (X.reshape(-1,1)-C2)) > 0
#     print(test1)
#     return test1 and test2

# data = []
# with open('ET/matches.txt', 'r') as file:
#     for line in file:
#         row = [float(x) for x in line.strip().split()]
#         data.append(row)

# src_pts = np.array([[x, y] for x, y, x_, y_ in data], dtype=np.float32)
# dst_pts = np.array([[x_, y_] for x, y, x_, y_ in data], dtype=np.float32)

# src_img = cv2.imread('ET/et1.jpg', cv2.IMREAD_COLOR)

# matches = np.hstack((src_pts, dst_pts))

# max_distance = 50
# plot_matches(src_img, matches, max_distance)

# ###### 1b ######

# inliers, H = ransac(matches, 0.5, 2000)
# plot_matches(src_img, inliers, 1000)

# print("\nH:")
# print(H)

# ###### 1c ######
# K = np.loadtxt('ET/K.txt')
# print("\nK:")
# print(K)
# E = K.T @ H @ K 
# print(E)

# U, S, Vt = np.linalg.svd(E)

# det_U = np.linalg.det(U)
# det_V = np.linalg.det(Vt.T)

# if (det_U > 0) and (det_V < 0):
#     E = -E
#     Vt = - Vt

# elif (det_U < 0) and (det_V > 0):
#     E = -E
#     U = -U
# print("Matrix U:")
# print(U)

# print("\nSingular values (S):")
# print(S)

# print("\nMatrix V^T:")
# print(Vt)

# ###### 1d ######
# # Calculate P
# I_O = np.array([[1, 0, 0, 0], 
#        [0, 1, 0, 0],
#        [0, 0, 1, 0]])

# P = K @ I_O
# print("\nP:")
# print(P)

# # Calculate P'
# W = np.array([[1, -1, 0],
#               [1, 0, 0],
#               [0, 0, 1]])
# u3 = U[:,2]

# UWVt = U @ W @ Vt
# UWtVt = U @ W.T @ Vt

# u3 = u3.reshape(-1,1)
# x1 = np.hstack([UWVt, u3])
# x2 = np.hstack([UWVt, -u3])
# x3 = np.hstack([UWtVt, u3])
# x4 = np.hstack([UWtVt, -u3])
# #print("\nx:")
# #print(x)

# P1 = K @ x1
# P2 = K @ x2
# P3 = K @ x3
# P4 = K @ x4

# n1 = np.eye(3) @ np.array([[0],[0],[1]])
# n2_1 = UWVt @  np.array([[0],[0],[1]])
# n2_2 = UWtVt @  np.array([[0],[0],[1]])

# C1 = np.array([[0],[0],[0]])

# K2_1, R2_1, C2_1 = decomposeP(P1)
# K2_2, R2_2, C2_2 = decomposeP(P2)
# K2_3, R2_3, C2_3 = decomposeP(P3)
# K2_4, R2_4, C2_4 = decomposeP(P4)

# X = triangulate_point(P, P1, src_pts[0], dst_pts[0])
# front = check_chirality(X, n2_1, C1, n2_1, C2_1)
# print(front)

# X = triangulate_point(P, P2, src_pts[0], dst_pts[0])
# front = check_chirality(X, n2_1, C1, n2_2, C2_2)
# print(front)

# X = triangulate_point(P, P3, src_pts[0], dst_pts[0])
# front = check_chirality(X, n2_1, C1, n2_2, C2_3)
# print(front)

# X = triangulate_point(P, P4, src_pts[0], dst_pts[0])
# front = check_chirality(X, n2_1, C1, n2_2, C2_4)
# print(front)

import cv2
import matplotlib.pyplot as plt 
import numpy as np
from typing import List, Tuple
plt.rcParams['figure.figsize'] = [15, 15]

def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def filter_matches(matches: np.ndarray, max_distance: float) -> np.ndarray:
    return matches[np.linalg.norm(matches[:, :2] - matches[:, 2:], axis=1) <= max_distance]

def random_point(matches: np.ndarray, k: int = 4) -> np.ndarray:
    return matches[np.random.choice(len(matches), k, replace=False)]

def get_error(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    num_points = len(points)
    all_p1 = np.hstack((points[:, :2], np.ones((num_points, 1))))
    all_p2 = points[:, 2:]
    
    estimate_p2 = (H @ all_p1.T).T
    estimate_p2 = estimate_p2[:, :2] / estimate_p2[:, 2:]
    
    return np.sum((all_p2 - estimate_p2)**2, axis=1)

def homography(pairs: np.ndarray) -> np.ndarray:
    A = np.zeros((2*len(pairs), 9))
    for i, (x1, y1, x2, y2) in enumerate(pairs):
        A[2*i] = [-x1, -y1, -1, 0, 0, 0, x2*x1, x2*y1, x2]
        A[2*i+1] = [0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2]
    
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)
    return H / H[2, 2]

def plot_matches(src_img: np.ndarray, matches: np.ndarray, max_distance: float):
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    filtered_matches = filter_matches(matches, max_distance)
    #filtered_matches = matches 
    
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(gray_img, cmap='gray')
    
    for x1, y1, x2, y2 in filtered_matches:
        ax.plot([x1, x2], [y1, y2], 'r-', linewidth=0.5)
        ax.plot(x1, y1, 'bo', markersize=5)
    
    ax.set_title(f'Feature Matches (Max Distance: {max_distance:.2f})')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def ransac(matches: np.ndarray, threshold: float, iters: int) -> Tuple[np.ndarray, np.ndarray]:
    best_inliers = np.array([])
    best_H = None
    max_inliers = 0
    
    for _ in range(iters):
        points = random_point(matches)
        H = homography(points)
        
        if np.linalg.matrix_rank(H) < 3:
            continue
        
        errors = get_error(matches, H)
        inliers = matches[errors < threshold]
        
        if len(inliers) > max_inliers:
            best_inliers = inliers
            max_inliers = len(inliers)
            best_H = homography(best_inliers)

    print(f"inliers/matches: {max_inliers}/{len(matches)}")
    return best_inliers, best_H

def find_homography(pts_src: np.ndarray, pts_dst: np.ndarray) -> np.ndarray:
    n = pts_src.shape[0]
    A = np.zeros((2*n, 12))
    
    for i, ((x, y), (X, Y, Z)) in enumerate(zip(pts_src, pts_dst)):
        A[2*i] = [0, 0, 0, 0, -X, -Y, -Z, -1, y*X, y*Y, y*Z, y]
        A[2*i+1] = [X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x]
    
    _, _, Vh = np.linalg.svd(A)
    return Vh[-1].reshape(3, 4)

def decomposeP(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    W = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    
    Qt, Rt = np.linalg.qr((W @ P[:, :3]).T)
    K = W @ Rt.T @ W
    R = W @ Qt.T
    
    D = np.diag(np.sign(np.diag(K)))
    K = K @ D
    R = D @ R
    
    c = -R.T @ np.linalg.inv(K) @ P[:, 3]
    
    return K, R, c

def triangulate_point(P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    A = np.vstack([
        x1[0] * P1[2] - P1[0],
        x1[1] * P1[2] - P1[1],
        x2[0] * P2[2] - P2[0],
        x2[1] * P2[2] - P2[1]
    ])
    
    _, _, V = np.linalg.svd(A)
    X = V[-1]
    return X[:3] / X[3]

def check_chirality(X: np.ndarray, n1: np.ndarray, C1: np.ndarray, n2: np.ndarray, C2: np.ndarray) -> bool:
    test1 = np.dot(n1.T, (X - C1)) > 0
    test2 = np.dot(n2.T, (X - C2)) > 0
    return bool(test1 and test2)

def main():
    # Load data
    data = np.loadtxt('ET/matches.txt')
    src_pts, dst_pts = data[:, :2], data[:, 2:]
    
    src_img = cv2.imread('ET/et1.jpg', cv2.IMREAD_COLOR)
    matches = np.hstack((src_pts, dst_pts))
    
    # Plot matches
    plot_matches(src_img, matches, max_distance=50)
    
    # RANSAC
    inliers, H = ransac(matches, threshold=0.5, iters=2000)
    plot_matches(src_img, inliers, max_distance=1000)
    print("\nHomography matrix H:")
    print(H)
    
    # Load camera matrix K
    K = np.loadtxt('ET/K.txt')
    print("\nCamera matrix K:")
    print(K)
    
    # Calculate Essential matrix E
    E = K.T @ H @ K
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
        
        
        print("\n X:")
        print(X)
        
        print("\n n1.T:")
        print(n1.T)
        
        print("\n C1:")
        print(C1)
        
        print("\n n2.T:")
        print(n2.T)
        
        print("\n C2:")
        print(C2.reshape(-1, 1))
        
        if check_chirality(X.reshape(-1, 1), n1, C1, n2, C2.reshape(-1, 1)):
            print(f"\nValid P' found (option {i+1}):")
            print(P_prime)
            break
    else:
        print("No valid P' found satisfying the chirality condition.")

if __name__ == "__main__":
    main()