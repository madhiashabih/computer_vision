import os
from PIL import Image
import numpy as np

def transform(input, output, target_size=(150, 210)):
    if not os.path.exists(output):
        os.makedirs(output)

    image_data = []
    labels = []
    
    for image in os.listdir(input):
        if image.lower().endswith(('.png', '.jpg', '.jpeg')):
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
