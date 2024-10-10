import numpy as np 
import argparse
from helper import transform, feature_vector, plot_knn, read_sift_descriptors, kmeans
import numpy as np
from sklearn.cluster import KMeans
import joblib

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
    
    input = "in/q1/train_set_1"
    output = "out/train_set_1/"

    fhat_1, U_alpha = feature_vector(input, output, None)

    # Convert all the images to feature vectors. 
    
    input = "in/q1/display"
    output = "out/display/"
    fhat_2, _ = feature_vector(input, output, U_alpha)
    
    input = "in/q1/train_set_2"
    output = "out/train_set_2/"
    fhat_2, _ = feature_vector(input, output, U_alpha)
    
    input = "in/q1/test_set"
    output = "out/test_set/"
    X_test, _ = feature_vector(input, output, U_alpha)
    
    X_train_list = fhat_1 + fhat_2
    print(f"train_set: ")
    print("Number of elements in train_set:", len(X_train_list))  # Should be 500
    print("Shape of each train_set matrix:", X_train_list[0].shape)  # Should be (67500, 1)
    
    y_train = np.repeat(np.arange(50),5)
    y_train = np.concatenate((y_train, y_train))
    print("Updated y_train size after doubling:", y_train.shape) 
    
    y_test = np.repeat(np.arange(50),5)
    print("y_test size:", y_test.shape)
    
    # Use a kNN classifier to identify the test vectors, and plot accuracy as a function of the hyperparameter k ∈ {1,2,...,10}
    # Create 2D array
    X_train_columns = np.hstack(X_train_list)
    
    X_train = X_train_columns.T
    
    print("X_train")
    print(f"X train shape {X_train.shape}")
    
    # Step 1: Perform K-means Clustering
    inertias = []

    for k in range(1, 11):  # Testing k from 1 to 10
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X_train)
        inertias.append(kmeans.inertia_)  # Store the inertia

    plot_knn(inertias)

def q2():
    print("Running function q2...")

    # Detect SIFT features in all images from the training set
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

    # Perform k-means clustering
    k = 50  # Number of clusters
    visual_words = kmeans(k, sift_descriptors)
    
    # Save the KMeans model
    model_path = "kmeans_model.joblib"
    joblib.dump(visual_words, model_path)
    print(f"KMeans model saved at {model_path}")

    print("Visual Words (Central Points):")
    print(visual_words)
    
    
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

