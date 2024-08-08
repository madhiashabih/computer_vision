# import numpy as np
# import cv2

# # Load images
# query_img = cv2.imread('semper1.jpg')
# train_img = cv2.imread('semper2.jpg')

# # Convert images to grayscale
# query_img_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
# train_img_gray = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

# # Initialize ORB detector
# orb = cv2.ORB_create()

# # Detect keypoints and compute descriptors
# queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_gray, None)
# trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_gray, None)

# # Match descriptors using BFMatcher
# matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = matcher.match(queryDescriptors, trainDescriptors)

# # Sort matches by distance (best matches first)
# matches = sorted(matches, key=lambda x: x.distance)

# # Draw first 20 matches
# # final_img = cv2.drawMatches(query_img, queryKeypoints, 
# #                            train_img, trainKeypoints, 
# #                            matches[:20], None, 
# #                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# for match in matches[:20]:
#     # Access the properties of the match
#     # print(f"Distance: {match.distance}")
#     # print(f"Queryidx: {match.queryIdx}")
#     # print(f"Trainidx: {match.trainIdx}")

#     cv2.line(train_img, match.trainIdx, match.queryIdx, (0, 255, 0), 2)
#     cv2.line(query_img, match.trainIdx, match.queryIdx, (0, 255, 0), 2)
        
# # Optionally, resize the output image
# #final_img = cv2.resize(final_img, (1000, 650))

# # Save the final image
# cv2.imwrite('output.jpg', train_img)

# # Display the image
# cv2.imshow('Matches', train_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import numpy as np
import cv2

# Load images
query_img = cv2.imread('semper1.jpg')
train_img = cv2.imread('semper2.jpg')

# Convert images to grayscale
query_img_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
train_img_gray = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

# Initialize ORB detector
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors
queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_gray, None)
trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_gray, None)

# Match descriptors using BFMatcher
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(queryDescriptors, trainDescriptors)

# Sort matches by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Draw the top 20 matches
for match in matches[:20]:
    # Get the keypoint locations
    (x1, y1) = trainKeypoints[match.trainIdx].pt
    (x2, y2) = queryKeypoints[match.queryIdx].pt

    # Draw the matches
    cv2.line(train_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.line(query_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

# Save the final image
cv2.imwrite('output.jpg', train_img)

# Display the image
cv2.imshow('Matches', train_img)
cv2.waitKey(0)
cv2.destroyAllWindows()