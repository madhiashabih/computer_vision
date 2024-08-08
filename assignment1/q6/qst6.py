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

# Initialize a list to store the accuracy for each scale factor
accuracy_list = []

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

    # Initialize correct match count
    correct_matches = 0

    # Calculate the match accuracy
    for match in matches[:100]:
        # Get the keypoint locations
        (x1, y1) = rescaleKeypoints[match.trainIdx].pt
        (x2, y2) = queryKeypoints[match.queryIdx].pt
        calculated = (x1 / scale, y1 / scale)
        if np.sqrt((x2 - calculated[0])**2 + (y2 - calculated[1])**2) < 5:
            correct_matches += 1

    accuracy = correct_matches / len(matches[:100]) * 100
    accuracy_list.append(accuracy)
    print(f"Accuracy at scale {scale:.1f}: {accuracy:.2f}%")

# Plot the accuracy as a function of scale factor
plt.plot(np.arange(0.1, 3.0, 0.1), accuracy_list)
plt.xlabel("Scale Factor")
plt.ylabel("Accuracy (%)")
plt.title("Match Accuracy vs Scale Factor")
plt.show()
