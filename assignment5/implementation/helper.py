import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def transform(input, output, target_size=(150, 210)):
    if not os.path.exists(output):
        os.makedirs(output)

    image_data = []
    labels = []
    
    for image in os.listdir(input):
        if image.lower().endswith(('.jpg')):
            image_path = os.path.join(input, image)
            with Image.open(image_path) as img:
                # Convert to grayscale
                img = img.convert('L')
                        
                # Resize the image
                img = img.resize(target_size, Image.LANCZOS)
                        
                # Save the preprocessed image
                output_path = os.path.join(output, f"{image}")
                img.save(output_path)
                        
                # Convert image to numpy array and flatten
                img_array = np.array(img).flatten()
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
    plt.grid(True)

    # Add rank information to the plot
    plt.axvline(x=rank, color='r', linestyle='--', label=f'Rank = {rank}')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show() 
