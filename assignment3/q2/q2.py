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
    _, _, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    return L.reshape(3, 4)

def plot_3D(dst_pts_1, dst_pts_2):
    x1, y1, z1 = dst_pts_1[:,0], dst_pts_1[:,1], dst_pts_1[:,2]
    x2, y2, z2 = dst_pts_2[:,0], dst_pts_2[:,1], dst_pts_2[:,2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x1, y1, z1, color='r', label='Lego 1')
    ax.scatter(x2, y2, z2, color='b', label='Lego 2')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.legend()

    plt.show()

def main():
    file_path = 'cv_3.xlsx'
    src_pts = read_excel_data(file_path, 'Sheet3')
    dst_pts_2 = read_excel_data(file_path, 'Sheet4') 
       
    print(dst_pts_2)

    P2 = find_homography(src_pts, dst_pts_2)
        
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
    dst_pts_1 = read_excel_data(file_path, 'Sheet2') 
        
    P1 = find_homography(src_pts, dst_pts_1)
        
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
    
    plot_3D(dst_pts_1, dst_pts_2)
    
if __name__ == "__main__":
    main()
