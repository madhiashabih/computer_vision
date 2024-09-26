import cv2
import matplotlib.pyplot as plt 
import numpy as np
from typing import List, Tuple
plt.rcParams['figure.figsize'] = [15, 15]
from mpl_toolkits.mplot3d import Axes3D
import random

def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def filter_matches(matches: np.ndarray, max_distance: float) -> np.ndarray:
    print("Shape of matches:", matches.shape)
    return matches[np.linalg.norm(matches[:, :2] - matches[:, 2:], axis=1) <= max_distance]

def random_points(A_matches, B_matches, k=8):
    idx = random.sample(range(len(A_matches)), k)
    A_points = [A_matches[i] for i in idx]
    B_points = [B_matches[i] for i in idx]

    return A_points, B_points


def sampson_distance(F, x1, x2):
    F = np.array(F)
    x1 = np.array(x1).reshape(3, 1)
    x2 = np.array(x2).reshape(3, 1)
    
    Fx1 = np.dot(F, x1)
    Ftx2 = np.dot(F.T, x2)
    
    x2tFx1 = np.dot(x2.T, Fx1)
    
    denom1 = Fx1[0]**2 + Fx1[1]**2
    denom2 = Ftx2[0]**2 + Ftx2[1]**2
    
    sampson_dist = x2tFx1**2 / (denom1 + denom2)
    sampson_dist = float(sampson_dist[0, 0])
    
    return sampson_dist

def fundamental(pair_1: np.ndarray, pair_2: np.ndarray) -> np.ndarray:
    A = []
    for (x, y), (x_prime, y_prime) in zip(pair_1, pair_2):
        A.append([x*x_prime, y*x_prime, x_prime, x*y_prime, y*y_prime, y_prime, x, y, 1])
    
    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    # Let F be the last column of V
    f = V[-1]
    # Pack the elements of f into matrix f_hat
    F = f.reshape((3, 3))
    
    # U_F, Sigma_F, V_T_F = np.linalg.svd(f_hat)

    # # Adjust singular values to force rank 2
    # Sigma_F[2] = 0

    # # Reconstruct F with rank 2
    # F = U_F @ np.diag(Sigma_F) @ V_T_F
    
    #H = V[-1].reshape(3, 3)
    #return H / H[2, 2]
    F = F / F[2, 2]
    
    return F
    
def plot_matches(src_img: np.ndarray, matches: np.ndarray, max_distance: float):
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(gray_img, cmap='gray')
    
    for x1, y1, x2, y2 in matches:
        ax.plot([x1, x2], [y1, y2], 'r-', linewidth=0.5)
        ax.plot(x1, y1, 'bo', markersize=5)
    
    ax.set_title(f'Feature Matches (Max Distance: {max_distance:.2f})')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    

def ransac(matches: np.ndarray, threshold: float, iters: int) -> Tuple[np.ndarray, np.ndarray]:
    best_inliers_1 = np.array([])
    best_inliers_2 = np.array([])
    best_F = None
    max_inliers = 0
    
    matches_1 = matches[:,:2]
    matches_2 = matches[:,2:]
    
    for _ in range(iters):
        points_1, points_2 = random_points(matches_1, matches_2, k=8)
        F = fundamental(points_1, points_2)
        
        inliers_1 = []
        inliers_2 = []
        
        for one, two in zip(matches_1, matches_2):
            one_homo = np.append(one, 1)
            two_homo = np.append(two, 1)
            distances = sampson_distance(F, one_homo, two_homo)
            if distances < threshold:
                inliers_1.append(one)
                inliers_2.append(two)
        
        if len(inliers_1) > max_inliers:
            best_inliers_1 = inliers_1.copy()
            best_inliers_2 = inliers_2.copy()
            max_inliers = len(inliers_1)
    
    best_F = fundamental(best_inliers_1, best_inliers_2) 

    print(f"inliers/matches: {max_inliers}/{len(matches)}")
    return best_inliers_1, best_inliers_2, best_F

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

def triangulate_point_array(P, P_prime, src_pts, dst_pts):
    
    num_points = src_pts.shape[0]
    X = np.zeros((num_points, 4))  # Homogeneous 3D points
    
    for i in range(num_points):
        x1, y1 = src_pts[i]
        x2, y2 = dst_pts[i]

        # Create the A matrix for the homogeneous equation system
        A = np.zeros((4, 4))
        A[0] = x1 * P[2] - P[0]
        A[1] = y1 * P[2] - P[1]
        A[2] = x2 * P_prime[2] - P_prime[0]
        A[3] = y2 * P_prime[2] - P_prime[1]

        # Solve using SVD
        _, _, V = np.linalg.svd(A)
        X_homogeneous = V[-1]
        
        # Normalize the 3D point (convert to inhomogeneous)
        X[i] = X_homogeneous / X_homogeneous[-1]
    
    return X # Return the inhomogeneous 3D points

def check_chirality(X: np.ndarray, n1: np.ndarray, C1: np.ndarray, n2: np.ndarray, C2: np.ndarray) -> bool:
    test1 = np.dot(n1.T, (X - C1)) > 0
    test2 = np.dot(n2.T, (X - C2)) > 0
    return bool(test1 and test2)

def plot_3d_points(X, P1, P2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the 3D points
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='b', marker='o', s=1)
    
    # Plot camera positions
    plot_camera(ax, P1, 'Camera 1', color='r')
    plot_camera(ax, P2, 'Camera 2', color='g')
    
    # Set equal scaling for the three axes
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Reconstruction')
    
    plt.show()

def plot_camera(ax, P, label, color):
    # Extract camera center and direction
    camera_center = -np.linalg.inv(P[:, :3]) @ P[:, 3]
    ax.scatter(camera_center[0], camera_center[1], camera_center[2], c=color, marker='^', label=label)
    # Optionally, draw the direction of the camera
    
def is_point_in_front(P, X):
    R = P[:, :3]
    C = -np.linalg.inv(R) @ P[:, 3]
    
    n = R.T @ np.array([0, 0, 1])
    
    return n.T @ (X[:3] - C) > 0
