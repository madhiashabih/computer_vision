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
    siftDetector = cv2.SIFT_create()
    kp, des = siftDetector.detectAndCompute(img, None)
    return kp, des

def plot_sift(gray, rgb, kp):
    tmp = rgb.copy()
    img = cv2.drawKeypoints(gray, kp, tmp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img

kp_left, des_left = SIFT(left_gray)
kp_right, des_right = SIFT(right_gray)

kp_left_img = plot_sift(left_gray, left_rgb, kp_left)
kp_right_img = plot_sift(right_gray, right_rgb, kp_right)

plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.imshow(kp_left_img)
plt.subplot(1, 2, 2)
plt.imshow(kp_right_img)
plt.show()

def matcher(kp1, des1, img1, kp2, des2, img2, threshold):
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < threshold*n.distance:
            good.append([m])

    matches = []
    for pair in good:
        matches.append(list(kp1[pair[0].queryIdx].pt + kp2[pair[0].trainIdx].pt))

    matches = np.array(matches)
    return matches     

matches = matcher(kp_left, des_left, left_rgb, kp_right, des_right, right_rgb, 0.5)
def plot_matches(matches, total_img):
    match_img = total_img.copy()
    offset = total_img.shape[1]/2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(match_img).astype('uint8')) #　RGB is integer type
    
    ax.plot(matches[:, 0], matches[:, 1], 'xr')
    ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr')
     
    ax.plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]],
            'r', linewidth=0.5)

    plt.show()
    
total_img = np.concatenate((left_rgb, right_rgb), axis=1)
plot_matches(matches, total_img) # Good mathces
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
            best_H = H.copy()
            
    print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
    return best_inliers, best_H

inliers, H = ransac(matches, 0.5, 2000)
plot_matches(inliers, total_img) # show inliers matches

