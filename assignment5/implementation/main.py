import os
from torchvision.transforms import transforms 
from helper import transform,  process_images, average_values, calculate_X 
# a) Resize and/or crop all the images to a fixed size.
# input = "in/cropped_faces/"
# output = "in/q1/faces_transformed/"
# image_data= transform(input, output)

# print(f"Preprocessed {len(image_data)} images.")
# print(f"Image data shape: {image_data.shape}")

# b) Select 5 random images of each person, and put them aside as a test set.

# c) Of the images remaining, select 5 random images of each person. Then use these 250 images to find
# an average vector a and basis Uα. Plot the singular values of the matrix X (defined in the lecture
# slides) in order to pick a suitable value of α.

#  Stack columns of an image into one long vector of length m = pq
input = "in/q1/train_set"
vectors = process_images(input)
print(vectors[1], vectors[2])
avg_value = average_values(vectors)

print("Sample of average vector (top-left corner):")
print(avg_value[1], avg_value[2])

X = calculate_X(vectors, avg_value)
print(X[1])

# Suppose we have n image vectors f i, i = 1,...,n, with average a = 1 n n i=1 f i


# d) Reconstruct a few of the images from their feature vector representations (y in the lecture slides) and
# display them next to the originals, for some idea of how effective your dimensionality reduction is.



# e) Convert all the images to feature vectors. You should use the same a and Uα from part (c), found with
# 5 images of each person. But now, for the purposes of classification, there will be 10 training vectors
# and 5 test vectors per person. Use a kNN classifier to identify the test vectors, and plot accuracy as
# a function of the hyperparameter k ∈ {1,2,...,10}
