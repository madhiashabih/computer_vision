import os
import numpy as np 
from helper import transform, process_images, average_values, calculate_X, calculate_x, find_svd, plot_singular, get_U_alpha 
########## a) Resize and/or crop all the images to a fixed size. ##########
# input = "in/cropped_faces/"
# output = "in/q1/faces_transformed/"
# image_data= transform(input, output)

# print(f"Preprocessed {len(image_data)} images.")
# print(f"Image data shape: {image_data.shape}")

########### b) Select 5 random images of each person, and put them aside as a test set. ###########

########### c) Of the images remaining, select 5 random images of each person. Then use these 250 images to find
# an average vector a and basis Uα. Plot the singular values of the matrix X (defined in the lecture
# slides) in order to pick a suitable value of α. ############

input = "in/q1/train_set"

# Stack column of images into one long vector
vectors = process_images(input)
print(f"Stacked columns of matrix into one long vector, vectors[1]: {vectors[1]}")

# Find average
avg_value = average_values(vectors)
print(f"Average values: {avg_value}")

# Calculate xi
x = calculate_x(vectors, avg_value)
print(f"x: {x}")
print(f"x rows, columns: {x.shape[0]} {x.shape[1]}")

# Calculate X
X = calculate_X(vectors, x)
print(f"X: {X}")
print(f"X rows, columns: {X.shape[0]} {X.shape[1]}")

# Find basis U_a
rounded_X = 
U, s, VT = find_svd(X)
U_alpha = get_U_alpha(U, 2)
print(f"U alpha: {U_alpha}")
#plot_singular(X[1])

########### d) Reconstruct a few of the images from their feature vector representations (y in the lecture slides) and
# display them next to the originals, for some idea of how effective your dimensionality reduction is. ##########



########## e) Convert all the images to feature vectors. You should use the same a and Uα from part (c), found with
# 5 images of each person. But now, for the purposes of classification, there will be 10 training vectors
# and 5 test vectors per person. Use a kNN classifier to identify the test vectors, and plot accuracy as
# a function of the hyperparameter k ∈ {1,2,...,10} #########
