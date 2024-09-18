import cv2
import matplotlib.pyplot as plt 
import numpy as np
import random
plt.rcParams['figure.figsize'] = [15, 15]
from applyhomography import applyhomography
from PIL import Image  

def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def filter_matches(matches, max_distance):
    return [match for match in matches if calculate_distance(*match) <= max_distance]

def random_point(matches, k=4):
    idx = random.sample(range(len(matches)), k)
    point = [matches[i] for i in idx ]
    return np.array(point)

def get_error(points, H):
    num_points = len(points)
    all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
    all_p2 = points[:, 2:4]
    estimate_p2 = np.zeros((num_points, 2))
    for i in range(num_points):
        temp = np.dot(H, all_p1[i])
        estimate_p2[i] = (temp/temp[2])[0:2] # set index 2 to 1 and slice the index 0, 1
    # Compute error
    errors = np.linalg.norm(all_p2 - estimate_p2 , axis=1) ** 2
    return errors

def homography(pairs):
    rows = []
    for i in range(pairs.shape[0]):
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)
        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)
    U, s, V = np.linalg.svd(rows)
    H = V[-1].reshape(3, 3)
    H = H/H[2, 2] # standardize to let w*H[2,2] = 1
    return H

def plot_matches(src_img, matches, max_distance):
    # Convert the source image to grayscale
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    
    filtered_matches = filter_matches(matches, max_distance)
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Display the grayscale image
    ax.imshow(gray_img, cmap='gray')
    
    # Plot the matches
    for match in filtered_matches:
        x1, y1, x2, y2 = match
        ax.plot([x1, x2], [y1, y2], 'r-', linewidth=0.5)
        ax.plot(x1, y1, 'bo', markersize=5)  # Blue circle for xi
    
    ax.set_title(f'Feature Matches on Grayscale Image (Max Distance: {max_distance:.2f})')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def ransac(matches, threshold, iters):
    num_best_inliers = 0
    
    for i in range(iters):
        points = random_point(matches)
        H = homography(points)
        
        #  avoid dividing by zero 
        if np.linalg.matrix_rank(H) < 3:
            continue
            
        errors = get_error(matches, H)
        idx = np.where(errors < threshold)[0]
        inliers = matches[idx]

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            #best_H = H.copy()
            best_H = homography(best_inliers)

    print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
    return best_inliers, best_H

def find_homography(pts_src: np.ndarray, pts_dst: np.ndarray) -> np.ndarray:
    """Calculate the homography matrix using SVD."""
    n = pts_src.shape[0]
    A = np.zeros((2*n, 12))
    
    for i in range(n):
        x, y = pts_src[i]
        X, Y, Z = pts_dst[i]
        A[2*i] = [0, 0, 0, 0, -X, -Y, -Z, -1, y*X, y*Y, y*Z, y]
        A[2*i+1] = [X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x]
    
    _, _, Vh = np.linalg.svd(A)
    return Vh[-1].reshape(3, 4)

def decomposeP(P):
    '''
        The input P is assumed to be a 3-by-4 homogeneous camera matrix.
        The function returns a homogeneous 3-by-3 calibration matrix K,
        a 3-by-3 rotation matrix R and a 3-by-1 vector c such that
        K*R*[eye(3), -c] = P.

    '''

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

# Read the data from the text file
data = []
with open('ET/matches.txt', 'r') as file:
    for line in file:
        row = [float(x) for x in line.strip().split()]
        data.append(row)

# Extract the (x, y) and (x', y') coordinates
src_pts = np.array([[x, y] for x, y, x_, y_ in data], dtype=np.float32)
dst_pts = np.array([[x_, y_] for x, y, x_, y_ in data], dtype=np.float32)

# Load the source and destination images
src_img = cv2.imread('ET/et1.jpg', cv2.IMREAD_COLOR)

matches = np.hstack((src_pts, dst_pts))

# Create a combined image with the source and destination images side-by-side
max_distance = 50
plot_matches(src_img, matches, max_distance)

###### 1b ######

inliers, H = ransac(matches, 0.5, 2000)
plot_matches(src_img, inliers, 1000)

print("\nH:")
print(H)

###### 1c ######
K = np.loadtxt('ET/K.txt')
E = K.T @ H @ K 
print(E)

# Perform Singular Value Decomposition
U, S, Vt = np.linalg.svd(E)

det_U = np.linalg.det(U)
det_V = np.linalg.det(Vt.T)

if (det_U > 0) and (det_V < 0):
    E = -E
    Vt = - Vt

elif (det_U < 0) and (det_V > 0):
    E = -E
    U = -U
print("Matrix U:")
print(U)

print("\nSingular values (S):")
print(S)

print("\nMatrix V^T:")
print(Vt)

###### 1d ######

I = np.eye(3)
column = np.array([[0], [0], [0]])
P = np.dot(K, np.hstack((I, column))) 
print("\nP")
print(P)
# Calculate R's
R1 = np.array([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]])
C1 = np.array([0, 0, 0])

n1 = np.dot(R1.T, [0, 0, 1])

R2 = np.dot(U, np.dot(S, Vt))
n2 = np.dot(R2, [0, 0, 1])

#K2, R2, C2 = decomposeP(H)

print("###### 1d ######")
print("R1:")
print(R1)
print("\nC1:")
print(C1)
print("\nn1:")
print(n1)

