import cv2
import matplotlib.pyplot as plt 
import numpy as np
from typing import List, Tuple
plt.rcParams['figure.figsize'] = [15, 15]
from mpl_toolkits.mplot3d import Axes3D

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
    
    return filtered_matches

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
    
    return X[:, :3]  # Return the inhomogeneous 3D points

def check_chirality(X: np.ndarray, n1: np.ndarray, C1: np.ndarray, n2: np.ndarray, C2: np.ndarray) -> bool:
    test1 = np.dot(n1.T, (X - C1)) > 0
    test2 = np.dot(n2.T, (X - C2)) > 0
    return bool(test1 and test2)

def plot_3D(X: np.ndarray, R1: np.ndarray, R2: np.ndarray, c1: np.ndarray, c2: np.ndarray):
    """Plot 3D points and camera orientations."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    ax.scatter(*X.T, color='r', label='Lego 1')
    
     # Define unique colors for each camera axis
    camera_axis_colors = {
        'Camera 1': ['g', 'c', 'm'],  # Green, Cyan, Magenta for Camera 1 (x, y, z axes)
        'Camera 2': ['y', 'orange', 'purple']  # Yellow, Orange, Purple for Camera 2 (x, y, z axes)
    }

    # Define labels for cameras
    camera_labels = ['Camera 1', 'Camera 2']
    axis_labels = ['x', 'y', 'z']  # Axis labels

    # Plot camera orientations
    for idx, (c, R) in enumerate([(c1, R1), (c2, R2)]):
        camera_label = camera_labels[idx]
        for i, axis in enumerate(axis_labels):
            b = np.dot(R.T, np.eye(3)[i])
            print(b)
            print(c)
            direction = np.dot(R.T, np.eye(3)[i]) * 500 + c.reshape(-1,1)
            
            # Use specific color for each axis of each camera
            color = camera_axis_colors[camera_label][i]
            ax.plot(*zip(c, direction), color=color, label=f'{camera_label} {axis.upper()}')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.legend()
    plt.show()

def main():
    # Load data
    data = np.loadtxt('fountain/matches.txt')
    src_pts, dst_pts = data[:, :2], data[:, 2:]
    
    src_img = cv2.imread('fountain/fountain1.jpg', cv2.IMREAD_COLOR)
    matches = np.hstack((src_pts, dst_pts))
    
    # Plot matches
    filtered_matches = plot_matches(src_img, matches, max_distance=400)
    
    # RANSAC
    inliers, H = ransac(filtered_matches, threshold=0.5, iters=2000)
    plot_matches(src_img, inliers, max_distance=1000)
    print("\nHomography matrix H:")
    print(H)
    
    # Load camera matrix K
    K1 = np.loadtxt('fountain/K1.txt')
    K2 = np.loadtxt('fountain/K2.txt')
    print("\nCamera matrix K:")
    print(K1)
    
    # Calculate Essential matrix E
    E = K1.T @ H @ K1
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
    P = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    print("\nProjection matrix P:")
    print(P)
    
    W = np.array([[1, -1, 0], [1, 0, 0], [0, 0, 1]])
    u3 = U[:, 2].reshape(-1, 1)
    
    potential_P_primes = [
        K2 @ np.hstack([U @ W @ Vt, u3]),
        K2 @ np.hstack([U @ W @ Vt, -u3]),
        K2 @ np.hstack([U @ W.T @ Vt, u3]),
        K2 @ np.hstack([U @ W.T @ Vt, -u3])
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
            # X = triangulate_point_array(P, P_prime, src_pts, dst_pts)
            
            break
    else:
        print("No valid P' found satisfying the chirality condition.")

if __name__ == "__main__":
    main()