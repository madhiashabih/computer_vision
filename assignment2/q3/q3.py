import numpy as np
from scipy import linalg
import scipy.ndimage
from applyhomography import applyhomography
import cv2

def find_homography(pts_src, pts_dst):
    A = []
    for i in range(4):
        x, y = pts_src[i][0], pts_src[i][1]
        u, v = pts_dst[i][0], pts_dst[i][1]
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    A = np.array(A)
    _, _, Vh = np.linalg.svd(A)
    L = Vh[-1, :]  / Vh[-1, -1]
    return L.reshape(3, 3)

img = cv2.imread('bricks.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h, w, _ = img.shape

# output = np.full((500, w, 3), 255, dtype=np.uint8)

# Define the points
# pts_src = np.array([[242, 344], [41, 485], [554, 368], [525, 544]])
pts_src = np.array([[344,499],[30,721],[847,531],[789,824]])
pts_dst = np.array([[0, 0], [0, 900], [900, 0], [900, 900]])

# Compute the homography
H = find_homography(pts_src, pts_dst)

output = applyhomography(img, H, img)

output = cv2.imwrite('output.jpg', cv2.cvtColor(output, cv2.COLOR_RGB2BGR))


