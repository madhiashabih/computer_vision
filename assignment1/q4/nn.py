import numpy as np
from PIL import Image

def nearest_neighbor_resize(image, scale_factor):
    # Get original image dimensions
    height, width, channels = image.shape
    
    # Calculate new dimensions
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    
    # Create an empty array for the resized image
    resized = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    
    # Compute the scaling ratios
    x_ratio = width / new_width
    y_ratio = height / new_height
    
    for i in range(new_height):
        for j in range(new_width):
            # Find the nearest pixel from the original image
            px = min(width - 1, int(j * x_ratio))
            py = min(height - 1, int(i * y_ratio))
            
            # Copy the pixel value
            resized[i, j] = image[py, px]
    
    return resized

# Example usage
if __name__ == "__main__":
    # Load an image
    input_image = np.array(Image.open("fruits.jpeg"))
    
    # Resize the image
    scale_factor = 3 
    resized_image = nearest_neighbor_resize(input_image, scale_factor)
    
    # Save the resized image
    Image.fromarray(resized_image).save("output_nn_{:.2f}.jpg".format(scale_factor))
    
    print(f"Original size: {input_image.shape}")
    print(f"Resized to: {resized_image.shape}")


