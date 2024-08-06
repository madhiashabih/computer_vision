import numpy as np
from PIL import Image
import cv2
import math 
import matplotlib.pyplot as plt

# Source: https://meghal-darji.medium.com/implementing-bilinear-interpolation-for-image-resizing-357cbb2c2722

def detect_features(image, nfeatures=500):
    """
    Detect ORB features in an image.
    
    Args:
    image_path (str): Path to the input image.
    nfeatures (int): Number of features to detect.
    
    Returns:
    tuple: (keypoints, descriptors)
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(desc1, desc2, threshold=0.7):
    """
    Match features between two sets of descriptors.
    
    Args:
    desc1, desc2 (np.array): Descriptors from two images.
    threshold (float): Ratio test threshold for good matches.
    
    Returns:
    list: Good matches.
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:int(len(matches) * threshold)]
    return good_matches

def visualize_matches(img1, img2, kp1, kp2, matches):
    """
    Visualize matching features between two images.
    
    Args:
    img1_path, img2_path (str): Paths to the input images.
    kp1, kp2 (list): Keypoints from two images.
    matches (list): List of good matches.
    """
    
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    plt.figure(figsize=(20,10))
    plt.imshow(match_img)
    plt.axis('off')
    plt.show()

def bl_resize(original_img, new_h, new_w):
	#get dimensions of original image
	old_h, old_w, c = original_img.shape
	#create an array of the desired shape. 
	#We will fill-in the values later.
	resized = np.zeros((new_h, new_w, c))
	#Calculate horizontal and vertical scaling factor
	w_scale_factor = (old_w ) / (new_w ) if new_h != 0 else 0
	h_scale_factor = (old_h ) / (new_h ) if new_w != 0 else 0
	for i in range(new_h):
		for j in range(new_w):
			#map the coordinates back to the original image
			x = i * h_scale_factor
			y = j * w_scale_factor
			#calculate the coordinate values for 4 surrounding pixels.
			x_floor = math.floor(x)
			x_ceil = min( old_h - 1, math.ceil(x))
			y_floor = math.floor(y)
			y_ceil = min(old_w - 1, math.ceil(y))

			if (x_ceil == x_floor) and (y_ceil == y_floor):
				q = original_img[int(x), int(y), :]
			elif (x_ceil == x_floor):
				q1 = original_img[int(x), int(y_floor), :]
				q2 = original_img[int(x), int(y_ceil), :]
				q = q1 * (y_ceil - y) + q2 * (y - y_floor)
			elif (y_ceil == y_floor):
				q1 = original_img[int(x_floor), int(y), :]
				q2 = original_img[int(x_ceil), int(y), :]
				q = (q1 * (x_ceil - x)) + (q2	 * (x - x_floor))
			else:
				v1 = original_img[x_floor, y_floor, :]
				v2 = original_img[x_ceil, y_floor, :]
				v3 = original_img[x_floor, y_ceil, :]
				v4 = original_img[x_ceil, y_ceil, :]

				q1 = v1 * (x_ceil - x) + v2 * (x - x_floor)
				q2 = v3 * (x_ceil - x) + v4 * (x - x_floor)
				q = q1 * (y_ceil - y) + q2 * (y - y_floor)

			resized[i,j,:] = q
	return resized.astype(np.uint8)

A = cv2.imread('semper1.jpg', cv2.IMREAD_COLOR)
k = 2
B = bl_resize(A, k * A.shape[0], k * A.shape[1])
cv2.imwrite('output.jpeg', B)

# Detect features
kp1, desc1 = detect_features(A, nfeatures=100)
kp2, desc2 = detect_features(B, nfeatures=100)
    
# Match features
good_matches = match_features(desc1, desc2, threshold=0.7)
    
# Visualize matches
visualize_matches(A, B, kp1, kp2, good_matches)

print(f"Number of matches: {len(good_matches)}")