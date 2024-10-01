import os
from torchvision.transforms import transforms 
from helper import transform
# a) Resize and/or crop all the images to a fixed size.
input = "in/cropped_faces/"
output = "in/cropped_faces_transformed/"
image_data= transform(input, output)

print(f"Preprocessed {len(image_data)} images.")
print(f"Image data shape: {image_data.shape}")

# b) Select 5 random images of each person, and put them aside as a test set.

# c) Of the images remaining, select 5 random images of each person. Then use these 250 images to find
# an average vector a and basis Uα. Plot the singular values of the matrix X (defined in the lecture
# slides) in order to pick a suitable value of α.

# d) Reconstruct a few of the images from their feature vector representations (y in the lecture slides) and
# display them next to the originals, for some idea of how effective your dimensionality reduction is.

# e) Convert all the images to feature vectors. You should use the same a and Uα from part (c), found with
# 5 images of each person. But now, for the purposes of classification, there will be 10 training vectors
# and 5 test vectors per person. Use a kNN classifier to identify the test vectors, and plot accuracy as
# a function of the hyperparameter k ∈ {1,2,...,10}
