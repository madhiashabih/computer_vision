import pandas as pd
import numpy as np
from decomposeP import decomposeP

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
    n = pts_src.shape[0]
    print("n: ")
    print(n) 

    A = []
    for i in range(n):
        x, y = pts_src[i][0], pts_src[i][1]
        X, Y, Z = pts_dst[i][0], pts_dst[i][1], pts_dst[i][2]
        A.append([0, 0, 0, 0, -X, -Y, -Z, -1, y*X, y*Y, y*Z, y])
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x])
    A = np.array(A)
    _, _, Vh = np.linalg.svd(A)
    return Vh[-1].reshape(3, 4)

def main():
    file_path = 'cv_3.xlsx'
    src_pts = read_excel_data(file_path, 'Sheet1')
    dst_pts = read_excel_data(file_path, 'Sheet2') 
        
    print(src_pts)
    print(dst_pts)

    P = find_homography(src_pts, dst_pts)
        
    print("\nHomography matrix P:")
    print(P)

    K, R, c = decomposeP(P)

    scaling_factor = K[2, 2]
    K_scaled = K / scaling_factor

    print("\nIntrinsic matrix K:")
    print(K_scaled)
    print("\nRotation matrix R:")
    print(R)
    print("\nCamera center C:")
    print(c)

if __name__ == "__main__":
    main()
