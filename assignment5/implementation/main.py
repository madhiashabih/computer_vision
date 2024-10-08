import os
import numpy as np 
import argparse
from helper import transform, process_images, average_values, calculate_X, calculate_x, find_svd, plot_singular, get_U_alpha, calculate_y, calculate_fhat, vector_to_image, kmeans, read_sift_descriptors

def q1():
    print("Running function q1...")
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
    U, s, VT = find_svd(X)
    U_alpha = get_U_alpha(U, 50)
    print(f"U alpha: {U_alpha}")
    plot_singular(s,X)

    ########### d) Reconstruct a few of the images from their feature vector representations (y in the lecture slides) and
    # display them next to the originals, for some idea of how effective your dimensionality reduction is. ##########
    y = calculate_y(U_alpha, vectors, 50)
    print(f"y: {y}")

    fhat = calculate_fhat(50, U_alpha, y)

    vector_to_image(fhat, (150, 210), "~/computer_vision/assignment5/reconstructed_image.png")

    ########## e) Convert all the images to feature vectors. You should use the same a and Uα from part (c), found with
    # 5 images of each person. But now, for the purposes of classification, there will be 10 training vectors
    # and 5 test vectors per person. Use a kNN classifier to identify the test vectors, and plot accuracy as
    # a function of the hyperparameter k ∈ {1,2,...,10} #########

def q2():
    print("Running function q2...")
    ##### Question 2 #####
    # 1. detect SIFT features in all images from the training set
    # Define the base folder and subfolders
    base_folder = "in/sift/"
    subfolders = [
        "Coast/train", "Forest/train", "Highway/train", "Kitchen/train",
        "Mountain/train", "Office/train", "Store/train", "Street/train", "Suburb/train"
    ]

    # Read all SIFT descriptors
    sift_descriptors = read_sift_descriptors(base_folder, subfolders)

    # Print the number of descriptors and the size of each descriptor
    print(f"Number of descriptors: {len(sift_descriptors)}")
    
    print(f"Size of each descriptor: {len(sift_descriptors[0])}")
    

    # Now you can perform k-means clustering
    # k = 50  # Number of clusters
    # visual_words = kmeans(k, descriptor_list)

    # print("Visual Words (Central Points):")
    # print(visual_words)
    # #  3. that’s our visual vocabulary fixed: a set of k 128D cluster centres
    #  4. we can now represent any image (in the training set or otherwise) as a normalised histo
    # gram of these words (why normalised?)
    
def main():
    parser = argparse.ArgumentParser(description="Run specific functions based on command-line arguments.")
    parser.add_argument("args", type=int, choices=[1, 2], help="Choose 1 to run q1 or 2 to run q2.")
    
    # Parse the arguments
    arguments = parser.parse_args()

    # Run the corresponding function
    if arguments.args == 1:
        q1()
    elif arguments.args == 2:
        q2()

if __name__ == "__main__":
    main()

