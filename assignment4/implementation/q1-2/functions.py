import cv2
import matplotlib.pyplot as plt 
import numpy as np
from typing import List, Tuple
plt.rcParams['figure.figsize'] = [15, 15]
import random
import numpy.linalg as la
from applyhomography import applyhomography

def load_data(matches_file: str, image_file: str, image_file_2: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load match data and images."""
    data = np.loadtxt(matches_file)
    src_pts, dst_pts = data[:, :2], data[:, 2:]
    src_img = cv2.imread(image_file, cv2.IMREAD_COLOR)
    dst_img = cv2.imread(image_file_2, cv2.IMREAD_COLOR)
    matches = np.hstack((src_pts, dst_pts))
    return matches, src_img, dst_img

def process_matches(matches: np.ndarray, src_img: np.ndarray, max_distance: float = 150) -> np.ndarray:
    """Process and plot matches."""
    filtered_matches = filter_matches(matches, max_distance=max_distance)
    plot_matches(src_img, filtered_matches, max_distance)
    return filtered_matches

def compute_fundamental_matrix(src_img, matches: np.ndarray, threshold: float = 1, iters: int = 2000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute fundamental matrix using RANSAC."""
    inliers_1, inliers_2, F = ransac(matches, threshold=threshold, iters=iters)
    inliers = np.hstack((inliers_1, inliers_2))
    plot_matches(src_img, inliers, max_distance=150)
    return inliers_1, inliers_2, F

def compute_essential_matrix(F: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute essential matrix and perform SVD."""
    E = K.T @ F @ K
    U, S, Vt = np.linalg.svd(E)
    
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E = -E
        U = -U if np.linalg.det(U) < 0 else U
        Vt = -Vt if np.linalg.det(Vt) < 0 else Vt
    
    return E, U, S, Vt

def compute_projection_matrices(K: np.ndarray, U: np.ndarray, Vt: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Compute projection matrices."""
    R = np.eye(3)
    P = K @ np.hstack([R, np.zeros((3, 1))])
    
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    u3 = U[:, 2].reshape(-1, 1)
    
    potential_P_primes = [
        K @ np.hstack([U @ W @ Vt, u3]),
        K @ np.hstack([U @ W @ Vt, -u3]),
        K @ np.hstack([U @ W.T @ Vt, u3]),
        K @ np.hstack([U @ W.T @ Vt, -u3])
    ]
    
    return P, potential_P_primes

def find_valid_projection_matrix(P: np.ndarray, potential_P_primes: List[np.ndarray], inliers_1: np.ndarray, inliers_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find valid projection matrix using chirality condition."""
    n1 = np.array([0, 0, 1]).reshape(-1, 1)
    C1 = np.zeros((3, 1))
    
    for P_possible_prime in potential_P_primes:
        X = triangulate_point(P, P_possible_prime, inliers_1[0], inliers_2[0])
        R_possible_prime = P_possible_prime[:,:3]
        C2 = -np.linalg.inv(R_possible_prime) @ P_possible_prime[:, 3]
        n2 = R_possible_prime.T @ np.array([0, 0, 1])
        
        if check_chirality(X.reshape(-1,1), n1, C1, n2.reshape(-1,1), C2.reshape(-1,1)):
            return P_possible_prime, R_possible_prime
    
    return None, None

def compute_rectification_transforms(P: np.ndarray, P_prime: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rectification transforms."""
    K1, R1, c1 = decomposeP(P)
    K2, R2, c2 = decomposeP(P_prime)
    
    Kn = 0.5 * (K1 + K2)
    k = R1.T @ np.array([0, 0, 1])
    
    r1 = (c2 - c1) / la.norm(c2 - c1)
    r2 = np.cross(k, r1) / la.norm(np.cross(k, r1))
    r3 = np.cross(r1, r2)
    Rn = np.vstack((r1, r2, r3))

    T1 = Kn @ Rn @ R1.T @ la.inv(K1)
    T2 = Kn @ Rn @ R2.T @ la.inv(K2)
    
    return T1, T2

def apply_rectification(src_img: np.ndarray, dst_img: np.ndarray, T1: np.ndarray, T2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply rectification transforms to images."""
    _, _, transformed_1 = applyhomography(src_img, T1)
    _, _, transformed_2 = applyhomography(dst_img, T2)
    return transformed_1, transformed_2

def save_images(transformed_1: np.ndarray, transformed_2: np.ndarray, output_path_1: str, output_path_2: str):
    """Save transformed images."""
    cv2.imwrite(output_path_1, transformed_1)
    cv2.imwrite(output_path_2, transformed_2)

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

def draw_camera(ax, C, R, label, colour='k', axis_length=1, shape='o'):
    axes = np.array([np.dot(R.T, [axis_length, 0, 0]) + C,
                     np.dot(R.T, [0, axis_length, 0]) + C,
                     np.dot(R.T, [0, 0, axis_length]) + C])

    # Fix y-axis.
    axes[1] = flip_point(C, axes[1])

    ax.scatter(C[0], C[1], C[2],
               marker=shape, color=colour, s=axis_length*10, label=f"Camera {label}")

    colours = ['r', 'g', 'b']
    labels = ["x-axis", "y-axis", "z-axis"]

    for i in range(3):
        ax.plot([C[0], axes[i][0]],
                [C[1], axes[i][1]],
                [C[2], axes[i][2]], 
                f"{colours[i]}-", linewidth=1, label=labels[i])


def draw_plot_3d(inliers1, inliers2, P1, P2, image1, image2, filename):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    
    _, R1, c1 = decomposeP(P1)
    _, R2, c2 = decomposeP(P2)

    for i in range(len(inliers1)):
        point3d = triangulate_point(P1, P2, inliers1[i], inliers2[i])
        bgr = image1[int(inliers1[i, 1]), int(inliers1[i, 0])]
        colour = [(bgr[2] / 255, bgr[1] / 255, bgr[0] / 255)]
        ax.scatter(point3d[0], point3d[1], point3d[2], c=colour, s=1)

    draw_camera(ax, c1, R1, 1, 'k', shape='^')
    draw_camera(ax, c2, R2, 2, 'k', shape='v')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2, 4)
    ax.set_zlim(0, 7)
    ax.set_aspect('equal')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    for h in [30,-30]:
        for v in [45, 135, 225, 315]:
            ax.view_init(h, v)
            plt.savefig(f"{filename}_{h}-{v}.pdf")
    # plt.show()
    plt.close()
    
def decomposeP(P):
    
    W = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]])

    # calculate K and R up to sign
    Qt, Rt = np.linalg.qr((W.dot(P[:,0:3])).T)
    K = W.dot(Rt.T.dot(W))
    R = W.dot(Qt.T)

    # correct for negative focal length(s) if necessary
    D = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    if K[0,0] < 0:
        D[0,0] = -1
    if K[1,1] < 0:
        D[1,1] = -1
    if K[2,2] < 0:
        D[2,2] = -1
    K = K.dot(D)
    R = D.dot(R)

    # calculate c
    c = -R.T.dot(np.linalg.inv(K).dot(P[:,3]))

    return K, R, c

def flip_point(origin, point):
    diff = origin - point
    flipped = point + 2 * diff

    return flipped
    

