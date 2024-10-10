import os
import numpy as np 
import argparse
from helper import transform, process_images, image_to_vector, average_values, calculate_X, calculate_x, find_svd, plot_singular, get_U_alpha, calculate_y, calculate_fhat, vector_to_image, kmeans, read_sift_descriptors
import joblib
import subprocess

def transforms():
    print("Running transforms...")
    
    # Resize and/or crop all the images to a fixed size.
    input = "in/cropped_faces/"
    output = "in/q1/faces_transformed/"
    image_data= transform(input, output)

    print(f"Preprocessed {len(image_data)} images.")
    print(f"Image data shape: {image_data.shape}")

    # Select 5 random images of each person, and put them aside as a test set.

    # RUN ./create_test_train.sh

def q1():
    print("Running function q1...")
    
    # Use these 250 images to find an average vector a and basis Uα. 
    
    input = "in/q1/train_set"

    # Stack column of images into one long vector
    vectors = process_images(input)
    print(f"vectors:")
    print(f"Size of column vector[1] (should be 22500): {vectors[1].shape[0]}")
    print("Number of elements in the list (should be 250):", len(vectors))

    # Find average
    avg_value = average_values(vectors)
    print()
    print(f"Size of average value vectors (should be 22500): {avg_value.shape[0]}")

    # Calculate xi
    x = calculate_x(vectors, avg_value)
    print()
    print("x:")
    print(f"Rows of x[1] (should be 22500): {x[1].shape[0]}")
    print(f"Columns of x[1] (should be 1): {x[1].shape[1]}")
    print("Number of elements in the list (should be 250):", len(x))

    # Calculate X
    X = calculate_X(x)
    print()
    print(f"X:")
    print(f"Size of X  : row (should be 22500): {X.shape[0]}, column (should be 250): {X.shape[1]}")
    

    # Find basis U_a
    U, s, VT = find_svd(X)
    U_alpha = get_U_alpha(U, 50)
    print()
    print(f"U alpha: ")
    print(f"Size of U_alpha  : row (should be ?): {U_alpha.shape[0]}, column (should be 50): {U_alpha.shape[1]}")
    
    # Plot the singular values of the matrix X in order to pick a suitable value of α.
    plot_singular(s,X)

    # Reconstruct a few of the images from their feature vector representations (y in the lecture slides) 
    
    y = calculate_y(U_alpha, vectors, 50)
    print()
    print(f"y: ")
    print(f"y rows, columns: {y.shape[0]} {y.shape[1]}")
    

    fhat = calculate_fhat(50, U_alpha, y)

    vector_to_image(fhat, (150, 210), "~/computer_vision/assignment5/reconstructed_image.png")

    # Display them next to the originals, for some idea of how effective your dimensionality reduction is.

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
    k = 50  # Number of clusters
    visual_words = kmeans(k, sift_descriptors)
    
    # Save the KMeans model
    model_path = "kmeans_model.joblib"
    joblib.dump(visual_words, model_path)
    print(f"KMeans model saved at {model_path}")

    print("Visual Words (Central Points):")
    print(visual_words)
    # #  3. that’s our visual vocabulary fixed: a set of k 128D cluster centres
    #  4. we can now represent any image (in the training set or otherwise) as a normalised histo
    # gram of these words (why normalised?)
    
    
    
def main():
    
    parser = argparse.ArgumentParser(description="Run specific functions based on command-line arguments.")
    parser.add_argument("args", type=int, choices=[0, 1, 2], help="Choose 0, 1 to run transforms, q1 or 2 to run q2.")
    
    # Parse the arguments
    arguments = parser.parse_args()

    # Run the corresponding function
    if arguments.args == 0:
        transforms()
    elif arguments.args == 1:
        q1()
    elif arguments.args == 2:
        q2()

if __name__ == "__main__":
    main()

