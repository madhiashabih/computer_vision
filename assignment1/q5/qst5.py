import numpy as np
import cv2

# Load images
query_img = cv2.imread('semper1.jpg')
train_img = cv2.imread('semper2.jpg')

# Convert images to grayscale
query_img_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
train_img_gray = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

# Initialize ORB detector
orb = cv2.SIFT_create()

# Detect keypoints and compute descriptors
queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_gray, None)
trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_gray, None)

# Match descriptors using BFMatcher
matcher = cv2.BFMatcher()
matches = matcher.match(queryDescriptors, trainDescriptors)

# Sort matches by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Create copies of the original images for visualization
query_img_vis = query_img.copy()
train_img_vis = train_img.copy()

# Draw the top 20 matches
for match in matches[:100]:
    # Get the keypoint locations
    (x1, y1) = trainKeypoints[match.trainIdx].pt
    (x2, y2) = queryKeypoints[match.queryIdx].pt
    
    cv2.drawKeypoints(train_img_vis, [trainKeypoints[match.trainIdx]], train_img_vis, color=(0, 0, 255))
    cv2.drawKeypoints(query_img_vis,  [queryKeypoints[match.queryIdx]], query_img_vis, color=(0, 0, 255))

    # Draw the matches
    cv2.line(train_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.line(query_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1, cv2.LINE_AA)

    cv2.circle(train_img, (int(x1), int(y1)), 2, (0, 0, 255), -1)
    cv2.circle(query_img, (int(x2), int(y2)), 2, (0, 0, 255), -1)
    
# Save the final image
cv2.imwrite('output_train_m.jpg', train_img)
cv2.imwrite('output_query_m.jpg', query_img)
cv2.imwrite('output_train_f.jpg', query_img_vis)
cv2.imwrite('output_query_f.jpg', train_img_vis)

