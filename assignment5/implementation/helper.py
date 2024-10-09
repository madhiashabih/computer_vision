import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.cluster import KMeans

def transform(input, output, target_size=(150, 150)):
    if not os.path.exists(output):
        os.makedirs(output)

    image_data = []
    labels = []
    
    for image in os.listdir(input):
        if image.lower().endswith(('.jpg')):
            image_path = os.path.join(input, image)
            with Image.open(image_path) as img:        
                # Resize the image
                img = img.resize(target_size, Image.LANCZOS)
                        
                # Save the preprocessed image
                output_path = os.path.join(output, f"{image}")
                img.save(output_path)
                        
                # Convert image to numpy array
                img_array = np.array(img)
                image_data.append(img_array)


    return np.array(image_data, dtype=np.float32)

def image_to_vector(image_path):
    img = Image.open(image_path)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    vector = np.array(img)
    # print(f"p*q of matrix: {vector.shape[0] * vector.shape[1]}")   
    vector = vector.flatten()
    return vector

def process_images(folder_path):
    vectors = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg')):
            image_path = os.path.join(folder_path, filename)
            # Stack columns of an image into one long vector m=pq
            vectors.append(image_to_vector(image_path))
    return np.array(vectors)

def average_values(vectors):
    return np.mean(vectors, axis=0)

def calculate_x(vectors, avg_value):
    x = vectors - avg_value
    return x

def calculate_X(vectors, x): 
    # Calculate X = (1/sqrt(n)) * x
    n = vectors.shape[0]  # number of vectors (columns)
    print(f"number of vectors (columns): {n}")
    X = (1 / np.sqrt(n)) * x
    
    return X
    
def find_svd(X):
    U, s, Vt = np.linalg.svd(X, full_matrices = False)
    return U, s, Vt

def get_U_alpha(U, alpha):
    if alpha > U.shape[1]:
        raise ValueError("α cannot be larger than the number of columns in U")
    return U[:, :alpha]

def plot_singular(s, X):
    rank = np.linalg.matrix_rank(X)
    print(f"Rank of X: {rank}")

    # Plot the singular values
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(s) + 1), s, 'bo-')
    plt.title('Singular Values of Matrix X')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.yscale('log')  # Use log scale for y-axis
    plt.ylim(10**2, 10**4)
    plt.grid(True)

    # Add rank information to the plot
    plt.axvline(x=rank, color='r', linestyle='--', label=f'Rank = {rank}')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.savefig("out/singular_values_plot.png")


def calculate_y(U_alpha, f, a):
    return U_alpha.T @ (f - a)

def calculate_fhat(a, U_alpha, y):
    return a + U_alpha @ y

def vector_to_image(vector, image_shape, output_path):
    """
    Converts a flattened vector back into an image with the specified shape.

    Args:
        vector (np.array): The flattened vector representing the image.
        image_shape (tuple): Shape of the original image (height, width, channels).
        output_path (str): Path to save the reconstructed image.

    Returns:
        None
    """
    # Reshape the vector into the original image shape
    img_array = vector.reshape(image_shape)

    # Convert the NumPy array back to an image
    img = Image.fromarray(img_array.astype('uint8'), 'RGB')

    # Save the reconstructed image
    img.save(output_path)
    print(f"Image saved at: {output_path}")
 
# Source: https://medium.com/@aybukeyalcinerr/bag-of-visual-words-bovw-db9500331b2f    
# A k-means clustering algorithm who takes 2 parameter which is number 
# of cluster(k) and the other is descriptors list(unordered 1d array)
# Returns an array that holds central points.

def read_sift_descriptors(base_folder, subfolders):
    all_descriptors = []
    
    for subfolder in subfolders:
        folder_path = os.path.join(base_folder, subfolder)
        file_pattern = os.path.join(folder_path, "*_descr.txt")
        
        for file_path in glob.glob(file_pattern):
            X = np.loadtxt(file_path)
            all_descriptors.append(X)
    return np.vstack(all_descriptors)


def kmeans(k, descriptor_list):
    """
    Performs K-means clustering on the descriptor list.
    
    Args:
    k (int): Number of clusters.
    descriptor_list (list): A list of descriptors.
    
    Returns:
    array: Central points (visual words) of the clusters.
    """
    # Convert descriptor_list to a 2D array for KMeans
    kmeans = KMeans(n_clusters=k, n_init=42)
    kmeans.fit(descriptor_list)
    visual_words = kmeans.cluster_centers_
    return visual_words




