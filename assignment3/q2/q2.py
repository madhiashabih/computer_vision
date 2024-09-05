# import pandas as pd
# import numpy as np
# from decomposeP import decomposeP
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from PIL import Image
# from matplotlib import image

# def read_excel_data(file_path, sheet_name):
#     """Read data from an Excel file and convert it to a numpy array."""
#     try:
#         sheet_data = pd.read_excel(file_path, sheet_name=sheet_name)
#         return sheet_data.to_numpy()
#     except FileNotFoundError:
#         print(f"Error: File '{file_path}' not found.")
#         return None
#     except Exception as e:
#         print(f"Error reading Excel file: {e}")
#         return None

# def find_homography(pts_src, pts_dst):
#     n = pts_src.shape[0] 

#     A = []
#     for i in range(n):
#         x, y = pts_src[i][0], pts_src[i][1]
#         X, Y, Z = pts_dst[i][0], pts_dst[i][1], pts_dst[i][2]
#         A.append([0, 0, 0, 0, -X, -Y, -Z, -1, y*X, y*Y, y*Z, y])
#         A.append([X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x])
#     A = np.array(A)
#     _, _, Vh = np.linalg.svd(A)
#     return Vh[-1].reshape(3, 4)

# def plot_3D(dst_pts_1, dst_pts_2, R1, R2, c1, c2):
#     x1, y1, z1 = dst_pts_1[:,0], dst_pts_1[:,1], dst_pts_1[:,2]
#     x2, y2, z2 = dst_pts_2[:,0], dst_pts_2[:,1], dst_pts_2[:,2]

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     Cx1 = np.dot(R1.T, np.array([1, 0, 0])) + c1
#     Cy1 = np.dot(R1.T, np.array([0,1,0])) + c1
#     Cz1 = np.dot(R1.T, np.array([0, 0, 1])) + c1 

#     Cx2 = np.dot(R2.T, np.array([1, 0, 0])) + c2
#     Cy2 = np.dot(R2.T, np.array([0,1,0])) + c2
#     Cz2 = np.dot(R2.T, np.array([0, 0, 1])) + c2     


#     # Plot the first set of points in red
#     ax.scatter(x1, y1, z1, color='r', label='Lego 1')

#     ax.scatter(x2, y2, z2, color='b', label='Lego 2')

#     # Plot the line using c1 as the origin
#     ax.plot([c1[0], Cx1[0]], [c1[1], Cx1[1]], [c1[2], Cx1[2]], 'r-', label='Cx1 Line')
#     ax.plot([c1[0], Cy1[0]], [c1[1], Cy1[1]], [c1[2], Cy1[2]], 'g-', label='Cy1 Line')
#     ax.plot([c1[0], Cz1[0]], [c1[1], Cz1[1]], [c1[2], Cz1[2]], 'b-', label='Cz1 Line')

#     # Plot the line using c1 as the origin
#     ax.plot([c2[0], Cx2[0]], [c2[1], Cx2[1]], [c2[2], Cx2[2]], 'r-', label='Cx2 Line')
#     ax.plot([c2[0], Cy2[0]], [c2[1], Cy2[1]], [c2[2], Cy2[2]], 'g-', label='Cy2 Line')
#     ax.plot([c2[0], Cz2[0]], [c2[1], Cz2[1]], [c2[2], Cz2[2]], 'b-', label='Cz2 Line')

#     ax.set_xlabel('X Axis')
#     ax.set_ylabel('Y Axis')
#     ax.set_zlabel('Z Axis')
#     ax.legend()

#     plt.show()

# def plot_lines(line, image_path):
#     # Open and convert the image to a numpy array
#     img = Image.open(image_path)
#     img_array = np.array(img)
#     width, height = img.size
#     data = image.imread(image_path)

#      # Extract line coefficients and plot each line
#     for i in range(line.shape[1]):
#         a, b, c = line[:, i]
#         print("a:", a)
#         print("b:", b)
#         print("c:", c)
        
#         # Compute the corresponding y values using the line equation
#         y_1 = -((a * width + c) / b)
#         y_2 = -(c / b)
        
#         print("y_1:", y_1)
#         print("y_2:", y_2)
        
#         # Plot the line
#         plt.plot((0, width), (y_2, y_1), color='red', linewidth=1)

#     # Plot the image
#     plt.imshow(data)
    
#     # Show plot
#     plt.axis('off')  # Optionally hide axes
#     plt.show()

# def skew_symmetric(a):

#     a1, a2, a3 = a
#     return np.array([
#         [0, -a3, a2],
#         [a3, 0, -a1],
#         [-a2, a1, 0]
#     ])

# def main():
#     file_path = 'cv_3.xlsx'
#     src_pts = read_excel_data(file_path, 'Sheet3')
#     dst_pts_2 = read_excel_data(file_path, 'Sheet4') 

#     P2 = find_homography(src_pts, dst_pts_2)
        
#     print("\nHomography matrix P:")
#     print(P2)

#     K2, R2, c2 = decomposeP(P2)

#     scaling_factor = K2[2,2]
#     K2_scaled = K2/scaling_factor 

#     print("\nIntrinsic matrix K2:")
#     print(K2_scaled)
#     print("\nRotation matrix R2:")
#     print(R2)
#     print("\nCamera center C2:")
#     print(c2)
    
#     file_path = 'cv_3.xlsx'
#     src_pts = read_excel_data(file_path, 'Sheet1')
#     dst_pts_1 = read_excel_data(file_path, 'Sheet2') 
        
#     P1 = find_homography(src_pts, dst_pts_1)
        
#     print("\nHomography matrix P:")
#     print(P1)

#     K1, R1, c1 = decomposeP(P1)

#     scaling_factor = K1[2,2]
#     K1_scaled = K1/scaling_factor

#     print("\nIntrinsic matrix K1:")
#     print(K1_scaled)
#     print("\nRotation matrix R1:")
#     print(R1)
#     print("\nCamera center C1:")
#     print(c1)
    
#     plot_3D(dst_pts_1, dst_pts_2, R1, R2, c1, c2)

#     ########## Question 4 ##########
#     # a
#     c1 = np.append(c1, 1)
#     c2 = np.append(c2, 1)

#     e1 = np.dot(P1, c2.T)
#     e2 = np.dot(P2, c1.T)
   
#     # Dehomogenize
#     e1 = e1 / e1[2]
#     e2 = e2 / e2[2]

#     print("e1: ")
#     print(e1)
#     print("e2: ")
#     print(e2)

#     # b
#     P_pinv = np.linalg.pinv(P1) # pseudo-inverse of P1
#     F = np.dot(skew_symmetric(e2), np.dot(P2, P_pinv))
#     print("F:")
#     print(F)
    
#     x = np.array([[1548, 1840, 1],
#        [1553, 1681, 1],
#        [1552, 1516, 1],
#        [1556, 1352, 1],
#        [1560, 1188, 1],
#        [1564, 1022, 1],
#        [1565,  851, 1]])

#     print("x.T[:,0]: ")
#     print(x.T[:,0])
    
#     line = np.dot(F, x.T)
#     print("l: ")
#     print(line)
#     path = 'lego2.jpg'
#     plot_lines(line, path)

# if __name__ == "__main__":
#     main()

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

    # Plot camera orientations
    for c, R, color in [(c1, R1, 'r'), (c2, R2, 'b')]:
        for i, axis in enumerate(['x', 'y', 'z']):
            direction = np.dot(R.T, np.eye(3)[i]) + c
            ax.plot(*zip(c, direction), f'{color}-', label=f'C{axis}{color[-1]}')

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

    F = skew_symmetric(e2) @ P2 @ np.linalg.pinv(P1)
    print(f"F:\n{F}")
    
    x = np.array([
        [1548, 1840, 1],
        [1553, 1681, 1],
        [1552, 1516, 1],
        [1556, 1352, 1],
        [1560, 1188, 1],
        [1564, 1022, 1],
        [1565,  851, 1]
    ])

    print(f"x.T[:,0]:\n{x.T[:,0]}")
    
    line = F @ x.T
    print(f"l:\n{line}")
    plot_lines(line, 'lego2.jpg')

if __name__ == "__main__":
    main()