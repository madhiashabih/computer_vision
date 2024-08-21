# import cv2
# import numpy as np
# from estimate_homography_ransac import estimate_homography_ransac
# # from applyhomography import a
# from plot_inliers import plot_inliers

# data = []
# with open('peroldsift12.txt', 'r') as file:
#     for line in file:
#         row = [float(x) for x in line.strip().split()]
#         data.append(row)

# src_pts = np.array([[x,y] for x, y, x_, y_ in data], dtype=np.float32)
# dst_pts = np.array([[x_,y_] for x, y, x_, y_ in data], dtype=np.float32)

# src_img = cv2.imread('perold1.jpg')
# dst_img = cv2.imread('perold2.jpg')

# combined_img = np.hstack((src_img, dst_img))

# for i, (x, y, x_, y_) in enumerate(data):
#     cv2.line(combined_img, (int(x), int(y)), (int(x_) + src_img.shape[1], int(y_)), (0, 255, 0), 1)

# cv2.imwrite('output.jpg', combined_img)

# H, inliers = estimate_homography_ransac(src_pts, dst_pts)

# for i in inliers:
#     x, y, x_, y_ = data[i]
#     cv2.line(combined_img, (int(x), int(y)), (int(x_) + src_img.shape[1], int(y_)), (0, 255, 0), 1)
    
# cv2.imwrite('output_2.jpg', combined_img)

import cv2
import matplotlib.pyplot as plt 
import numpy as np
import random 
from tqdm.notebook import tqdm
plt.rcParams['figure.figsize'] = [15, 15]

# Read image and convert it to gray 
def read_image(path):
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_gray, img, img_rgb
left_gray, left_origin, left_rgb = read_image('perold1.jpg')
right_gray, right_origin, right_rgb = read_image('perold2.jpg')

def SIFT(img):
    siftDetector = cv2.xfeatures2d.SIFT_create()
    
     