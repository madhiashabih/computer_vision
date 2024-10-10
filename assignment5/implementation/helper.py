import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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

def feature_vector(input, output, U_alpha):
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
    a = 50
    U, s, VT = find_svd(X)
    
    if U_alpha is None:
        U_alpha = get_U_alpha(U, a)
        print()
        print(f"U alpha: ")
        print(f"Size of U_alpha  : row (should be ?): {U_alpha.shape[0]}, column (should be 50): {U_alpha.shape[1]}")
    
    # Plot the singular values of the matrix X in order to pick a suitable value of α.
    plot_singular(s,X)

    # Reconstruct a few of the images from their feature vector representations (y in the lecture slides) 
    
    y = calculate_y(U_alpha, vectors, a)
    print()
    print(f"y: ")
    print("Number of y:", len(y))  # Should be 250
    print("Shape of each y matrix:", y[0].shape)  # Should be (50, 1)
    

    fhat = calculate_fhat(a, U_alpha, y)
    print()
    print(f"fhat: ")
    print("Number of fhat:", len(fhat))  # Should be 250
    print("Shape of each fhat matrix:", fhat[0].shape)  # Should be (22500, 1)


    # Display them next to the originals, for some idea of how effective your dimensionality reduction is.
    
    for i, f in enumerate(fhat):
        image = vector_to_image(f, 150, 150)
        image.save(f"{output}reconstructed_{i}.jpg")
        
    return fhat, U_alpha
        

def image_to_vector(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)  
    
    vector = img_array.flatten('F').reshape(-1,1)
    
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

def calculate_X(x): 
    # Calculate X = (1/sqrt(n)) * x
    n = len(x)  # number of vectors (columns)
    matrix = np.column_stack(x)
    scaling_factor = 1/np.sqrt(n)
    X = scaling_factor * matrix 
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
    results = []
    for x in f:
        # Compute (x - a)
        x_minus_a = x - a  # Shape: (22500, 1)
        
        # Compute transpose(U_a) * (x - a), resulting in shape (50, 1)
        result = U_alpha.T @ x_minus_a  # Matrix multiplication
        results.append(result)
     
    return results

def calculate_fhat(a, U_alpha, y):
    f_hat_list = []

    for y_i in y:
        # Compute f_hat = a + U_alpha @ y_i
        f_hat = a + U_alpha @ y_i  # Shape: (22500, 1)
        f_hat_list.append(f_hat)
        
    return f_hat_list

# def vector_to_image(vector, rows, cols):
#     # Reshape the vector back to a 2D array
#     img_array = vector.reshape((3, cols, rows)).transpose(2, 1, 0)
    
#     # Convert the array to uint8 type
#     img_array = img_array.astype(np.uint8)
    
#     # Create an image from the array
#     img = Image.fromarray(img_array, mode='RGB')
    
#     return img

def vector_to_image(vector, rows, cols):
    # Reshape the vector into a (rows, cols, 3) array (standard RGB format)
    img_array = vector.reshape((3, cols, rows)).transpose(2, 1, 0)
    
    # Convert the array to uint8 type (clip values between 0 and 255)
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    # Create an image from the array
    img = Image.fromarray(img_array, mode='RGB')
    
    return img

def plot_knn(inertias):
    # Step 2: Plot the Elbow method
    plt.figure()
    plt.plot(range(1, 11), inertias, marker='o')
    plt.title('K-means Clustering of Data')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.xticks(range(1, 11))  # Set x-ticks to range 1-10 for better readability
    plt.grid()
    plt.savefig("out/k-means.png")
    
 
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




