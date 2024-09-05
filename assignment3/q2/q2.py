import pandas as pd
import numpy as np
from decomposeP import decomposeP
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from matplotlib import image

def read_excel_data(file_path: str, sheet_name: str) -> np.ndarray:
    """Read data from an Excel file and convert it to a numpy array."""
    try:
        return pd.read_excel(file_path, sheet_name=sheet_name).to_numpy()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading Excel file: {e}")
    return None

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

def plot_3D(dst_pts_1: np.ndarray, dst_pts_2: np.ndarray, R1: np.ndarray, R2: np.ndarray, c1: np.ndarray, c2: np.ndarray):
    """Plot 3D points and camera orientations."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    ax.scatter(*dst_pts_1.T, color='r', label='Lego 1')
    ax.scatter(*dst_pts_2.T, color='b', label='Lego 2')
    
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
            direction = np.dot(R.T, np.eye(3)[i]) * 500 + c
            # Use specific color for each axis of each camera
            color = camera_axis_colors[camera_label][i]
            ax.plot(*zip(c, direction), color=color, label=f'{camera_label} {axis.upper()}')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.legend()
    plt.show()

def plot_lines(line: np.ndarray, image_path: str):
    """Plot epipolar lines on an image."""
    img = Image.open(image_path)
    width, height = img.size
    data = image.imread(image_path)

    plt.figure(figsize=(10, 8))
    for a, b, c in line.T:
        y1 = -((a * width + c) / b)
        y2 = -(c / b)
        plt.plot((0, width), (y2, y1), color='red', linewidth=1)

    plt.imshow(data)
    plt.axis('off')
    plt.show()

def skew_symmetric(a: np.ndarray) -> np.ndarray:
    """Create a skew-symmetric matrix from a 3D vector."""
    return np.array([
        [0, -a[2], a[1]],
        [a[2], 0, -a[0]],
        [-a[1], a[0], 0]
    ])

def process_camera(file_path: str, src_sheet: str, dst_sheet: str) -> tuple:
    """Process camera data and return calibration results."""
    src_pts = read_excel_data(file_path, src_sheet)
    dst_pts = read_excel_data(file_path, dst_sheet)
    
    P = find_homography(src_pts, dst_pts)
    K, R, c = decomposeP(P)
    
    K_scaled = K / K[2,2]
    
    print(f"\nHomography matrix P:\n{P}")
    print(f"\nIntrinsic matrix K:\n{K_scaled}")
    print(f"\nRotation matrix R:\n{R}")
    print(f"\nCamera center C:\n{c}")
    
    return P, K, R, c, dst_pts

def main():
    file_path = 'cv_3.xlsx'
    
    P2, K2, R2, c2, dst_pts_2 = process_camera(file_path, 'Sheet3', 'Sheet4')
    P1, K1, R1, c1, dst_pts_1 = process_camera(file_path, 'Sheet1', 'Sheet2')
    
    plot_3D(dst_pts_1, dst_pts_2, R1, R2, c1, c2)

    # Question 4
    c1 = np.append(c1, 1)
    c2 = np.append(c2, 1)

    e1 = P1 @ c2
    e2 = P2 @ c1
   
    e1 /= e1[2]
    e2 /= e2[2]

    print(f"e1:\n{e1}")
    print(f"e2:\n{e2}")

    F_1 = skew_symmetric(e2) @ P2 @ np.linalg.pinv(P1)
    print(f"F:\n{F_1}")
    
    x_1 = np.array([
        [1548, 1840, 1],
        [1553, 1681, 1],
        [1552, 1516, 1],
        [1556, 1352, 1],
        [1560, 1188, 1],
        [1564, 1022, 1],
        [1565,  851, 1]
    ])

    print(f"x.T[:,0]:\n{x_1.T[:,0]}")
    
    line = F_1 @ x_1.T
    print(f"l:\n{line}")
    plot_lines(line, 'lego2.jpg')

    x_2 = np.array([
        [989, 1770, 1],
        [991, 1625, 1],
        [992, 1478, 1],
        [992, 1329, 1],
        [993, 1179, 1],
        [995, 1027, 1],
        [996, 875, 1]
    ])
    
    F_2 = skew_symmetric(e1) @ P1 @ np.linalg.pinv(P2)
    line = F_2 @ x_2.T
    plot_lines(line, 'lego1.jpg')
    
if __name__ == "__main__":
    main()