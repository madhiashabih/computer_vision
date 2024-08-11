import numpy as np
import cv2
import matplotlib.pyplot as plt

def bl_resize(original_img, new_h, new_w):
    old_h, old_w, c = original_img.shape
    resized = np.zeros((new_h, new_w, c), dtype=np.uint8)
    w_scale = old_w / new_w
    h_scale = old_h / new_h

    for i in range(new_h):
        for j in range(new_w):
            x, y = i * h_scale, j * w_scale
            x0, x1 = int(np.floor(x)), min(old_h - 1, int(np.ceil(x)))
            y0, y1 = int(np.floor(y)), min(old_w - 1, int(np.ceil(y)))

            if x0 == x1 and y0 == y1:
                q = original_img[x0, y0]
            elif x0 == x1:
                q = original_img[x0, y0] * (y1 - y) + original_img[x0, y1] * (y - y0)
            elif y0 == y1:
                q = original_img[x0, y0] * (x1 - x) + original_img[x1, y0] * (x - x0)
            else:
                q1 = original_img[x0, y0] * (x1 - x) + original_img[x1, y0] * (x - x0)
                q2 = original_img[x0, y1] * (x1 - x) + original_img[x1, y1] * (x - x0)
                q = q1 * (y1 - y) + q2 * (y - y0)

            resized[i, j] = q

    return resized

def calculate_accuracy(matches, queryKeypoints, rescaleKeypoints, scale):
    correct_matches = 0

    for match in matches:
        # Get the keypoints from the match
        query_idx = match.queryIdx
        rescale_idx = match.trainIdx
        
        # Get the original and resized keypoints
        query_pt = queryKeypoints[query_idx].pt
        rescale_pt = rescaleKeypoints[rescale_idx].pt
        
        # Calculate the scaled keypoint position
        scaled_query_pt = (query_pt[0] * scale, query_pt[1] * scale)
        
        # Check if the match is correct (within a certain threshold)
        if np.linalg.norm(np.array(scaled_query_pt) - np.array(rescale_pt)) < 5:  # Threshold of 5 pixels
            correct_matches += 1
    
    accuracy = correct_matches / len(matches) if matches else 0
    return accuracy

# Load the images
query_img = cv2.imread('semper1.jpg')
train_img = cv2.imread('semper2.jpg')

# Convert to grayscale
query_img_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors in the original image
queryKeypoints, queryDescriptors = sift.detectAndCompute(query_img_gray, None)

# Create a BFMatcher
matcher = cv2.BFMatcher()

# Lists to store scale factors and corresponding accuracies
scales = []
accuracies = []

# Iterate over scale factors from 0.1 to 3.0 with a step of 0.1
for scale in np.arange(0.1, 3.0, 0.1):
    # Resize the image
    img_resized = bl_resize(query_img, int(query_img.shape[0] * scale), int(query_img.shape[1] * scale))

    # Detect keypoints and compute descriptors in the resized image
    rescaleKeypoints, rescaleDescriptors = sift.detectAndCompute(img_resized, None)

    # Match descriptors using BFMatcher
    matches = matcher.match(queryDescriptors, rescaleDescriptors)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Calculate accuracy
    accuracy = calculate_accuracy(matches, queryKeypoints, rescaleKeypoints, scale)
    
    # Store the scale and accuracy
    scales.append(scale)
    accuracies.append(accuracy)

    print(f"Scale: {scale:.1f}, Accuracy: {accuracy:.2%}")

# Visualize the top 10 matches
img_matches = cv2.drawMatches(query_img, queryKeypoints, img_resized, rescaleKeypoints, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,  singlePointColor=None, matchesMask=None, thickness=3)

# Convert BGR to RGB for displaying with matplotlib
img_matches_rgb = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)

# Display the matches
plt.figure(figsize=(12, 6))
plt.imshow(img_matches_rgb)
plt.title(f"Keypoint Matches at Scale Factor {scale}")
plt.show()  

# Plotting the accuracy against scale factors
plt.figure(figsize=(10, 6))
plt.plot(scales, accuracies, marker='o', linestyle='-', color='b')
plt.title('Accuracy vs Scale Factor')
plt.xlabel('Scale Factor')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

  
    	
   


