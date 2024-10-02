import os
from PIL import Image
import numpy as np

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
    vector = vector.flatten()
    return vector

def process_images(folder_path):
    vectors = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg')):
            image_path = os.path.join(folder_path, filename)
            vectors.append(image_to_vector(image_path))
    return vectors

def average_values(vectors):
    if not vectors:
        return None
    return np.mean(vectors, axis=0)

def calculate_X(vectors, avg_value):
    # Ensure vectors is a numpy array
    vectors = np.array(vectors)
    
    # Calculate x = vectors - avg_value
    x = vectors - avg_value
    
    # Calculate X = (1/sqrt(n)) * x
    n = vectors.shape[0]  # number of vectors (columns)
    print(f"number of vectors (columns): {n}")
    X = (1 / np.sqrt(n)) * x
    
    return X
    
    
