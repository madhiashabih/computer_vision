import cv2
import matplotlib.pyplot as plt 
import numpy as np
import random
plt.rcParams['figure.figsize'] = [15, 15]
from applyhomography import applyhomography
from PIL import Image 

# https://gist.github.com/tigercosmos/90a5664a3b698dc9a4c72bc0fcbd21f4
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

# Read the data from the text file
data = []
with open('peroldsift12.txt', 'r') as file:
    for line in file:
        row = [float(x) for x in line.strip().split()]
        data.append(row)

# Extract the (x, y) and (x', y') coordinates
src_pts = np.array([[x, y] for x, y, x_, y_ in data], dtype=np.float32)
dst_pts = np.array([[x_, y_] for x, y, x_, y_ in data], dtype=np.float32)

# Load the source and destination images
src_img = cv2.imread('perold1.jpg', cv2.IMREAD_COLOR)
#src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
dst_img = cv2.imread('perold2.jpg', cv2.IMREAD_COLOR)
#dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)

matches = np.hstack((src_pts, dst_pts))

# Create a combined image with the source and destination images side-by-side
combined_img = np.hstack((src_img, dst_img))
#plot_matches(matches, combined_img)

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
#plot_matches(inliers, combined_img) 

### Stitching ###
# Transform the destination image 
minx, miny, transformed = applyhomography(src_img, H)
transformed = cv2.convertScaleAbs(transformed)

minx = int(640 + minx)

crop_img = transformed[0:transformed.shape[0], 0:minx]

max_y = max(crop_img.shape[0], dst_img.shape[0])
max_x = crop_img.shape[1] + dst_img.shape[1]
stitched = np.zeros((max_y, max_x, 3), dtype=np.uint8)

y_offset = max(0, -int(miny))

#stitched[y_offset:y_offset+dst_img.shape[0], :dst_img.shape[1]] = dst_img
stitched[:transformed.shape[0], 0:transformed.shape[1]] = transformed
stitched[y_offset:y_offset+dst_img.shape[0], crop_img.shape[1]:] = dst_img
#stitched[:crop_img.shape[0], 0:crop_img.shape[1]] = crop_img


# Change black pixels to white
black_pixels = np.all(stitched == [0, 0, 0], axis=-1)
stitched[black_pixels] = [255, 255, 255]

cv2.imwrite("output/transformed.jpg", transformed)

cv2.imwrite("output/stitched.jpg", stitched)


