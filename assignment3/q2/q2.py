import pandas as pd
import numpy as np
from decomposeP import decomposeP
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_excel_data(file_path, sheet_name):
    """Read data from an Excel file and convert it to a numpy array."""
    try:
        sheet_data = pd.read_excel(file_path, sheet_name=sheet_name)
        return sheet_data.to_numpy()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

def find_homography(pts_src, pts_dst):
    A = []
    for i in range(4):
        x, y = pts_src[i][0], pts_src[i][1]
        X, Y, Z = pts_dst[i][0], pts_dst[i][1], pts_dst[i][2]
        A.append([0, 0, 0, 0, -X, -Y, -Z, 1, y*X, y*Y, y*Z, y])
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x])
    A = np.array(A)
    print(A)
    _, _, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    return L.reshape(3, 4)

def plot_3d_points_and_camera_axes(world_points, K1, R1, c1, K2, R2, c2):
    """Plot the 3D world points and camera axes."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot world points
    ax.scatter(world_points[:, 0], world_points[:, 1], world_points[:, 2], c='b', marker='o', label='World Points')

    # Plot camera 1 axes
    camera1_axes = R1.T  # Use the transpose to switch from camera to world coordinates
    ax.quiver(c1[0], c1[1], c1[2], camera1_axes[0, 0], camera1_axes[1, 0], camera1_axes[2, 0], color='r', label='Camera 1 X-axis')
    ax.quiver(c1[0], c1[1], c1[2], camera1_axes[0, 1], camera1_axes[1, 1], camera1_axes[2, 1], color='g', label='Camera 1 Y-axis')
    ax.quiver(c1[0], c1[1], c1[2], camera1_axes[0, 2], camera1_axes[1, 2], camera1_axes[2, 2], color='b', label='Camera 1 Z-axis')

    # Plot camera 2 axes
    camera2_axes = R2.T
    ax.quiver(c2[0], c2[1], c2[2], camera2_axes[0, 0], camera2_axes[1, 0], camera2_axes[2, 0], color='m', label='Camera 2 X-axis')
    ax.quiver(c2[0], c2[1], c2[2], camera2_axes[0, 1], camera2_axes[1, 1], camera2_axes[2, 1], color='y', label='Camera 2 Y-axis')
    ax.quiver(c2[0], c2[1], c2[2], camera2_axes[0, 2], camera2_axes[1, 2], camera2_axes[2, 2], color='c', label='Camera 2 Z-axis')

    # Set equal scaling for all axes
    ax.set_box_aspect([1, 1, 1])  # Ensures equal unit length along each axis
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()


def main():
    file_path = 'cv_3.xlsx'
    src_pts = read_excel_data(file_path, 'Sheet3')
    dst_pts = read_excel_data(file_path, 'Sheet4') 
        
    P2 = find_homography(src_pts, dst_pts)
        
    print("\nHomography matrix P:")
    print(P2)

    K2, R2, c2 = decomposeP(P2)

    scaling_factor = K2[2,2]
    K2_scaled = K2/scaling_factor 

    print("\nIntrinsic matrix K:")
    print(K2_scaled)
    print("\nRotation matrix R:")
    print(R2)
    print("\nCamera center C:")
    print(c2)
    
    file_path = 'cv_3.xlsx'
    src_pts = read_excel_data(file_path, 'Sheet1')
    dst_pts = read_excel_data(file_path, 'Sheet2') 
        
    P1 = find_homography(src_pts, dst_pts)
        
    print("\nHomography matrix P:")
    print(P1)

    K1, R1, c1 = decomposeP(P1)

    scaling_factor = K1[2,2]
    K1_scaled = K1/scaling_factor

    print("\nIntrinsic matrix K:")
    print(K1_scaled)
    print("\nRotation matrix R:")
    print(R1)
    print("\nCamera center C:")
    print(c1)
    
    # Assuming src_pts are world points
    world_points = read_excel_data(file_path, 'Sheet5')

    # Plot points and camera axes
    plot_3d_points_and_camera_axes(world_points, K1, R1, c1, K2, R2, c2)
    
if __name__ == "__main__":
    main()
