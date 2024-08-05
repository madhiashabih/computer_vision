# import numpy as np
# from PIL import Image
# import cv2
# import math 
# import matplotlib.pyplot as plt

# ### Take a colour image as input and scale (resize) it by a given factor s. ###

# # Read in image
# img = cv2.imread('fruits.jpeg', cv2.IMREAD_COLOR)

# k = 2

# height, width, channels = img.shape

# # Set the size of the new image
# col = img.shape[1] * k
# row = img.shape[0] * k 

# # Initialize scaled image
# result = np.zeros((k*height, k*width, channels), dtype=np.uint8)

# # Nearest Neighbour Interpolation
# for i in range(height):
#     for j in range(width):
#         for n in range(k):
#             result[k*i][k*j] = img[i][j]
#             for l in range (k*i - (k-1), k*i):
#                 for m in range (k*j - (k-1), k*j):
#                     print('l: {} and m: {}', l, m)
#                     result[l][m] = img[i][j]
#             # result[(k*i)-n][k*j] = img[i][j]
#             # result[k*i][(k*j)-n] = img[i][j]
#             # result[(k*i)-n][(k*j)-n] = img[i][j]

# cv2.imwrite('output_nn.jpeg', result)

import numpy as np
from PIL import Image

def nearest_neighbor_resize(image, scale_factor):
    """
    Resize a color image using nearest neighbor interpolation.
    
    Args:
    image (numpy.ndarray): Input color image as a numpy array (height, width, channels).
    scale_factor (float): Scale factor for resizing. Values > 1 enlarge, < 1 shrink the image.
    
    Returns:
    numpy.ndarray: Resized image as a numpy array.
    """
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
    input_image = np.array(Image.open("cat.jpeg"))
    
    # Resize the image
    scale_factor = 2 # Enlarge by 50%
    resized_image = nearest_neighbor_resize(input_image, scale_factor)
    
    # Save the resized image
    Image.fromarray(resized_image).save("resized_image.jpg")
    
    print(f"Original size: {input_image.shape}")
    print(f"Resized to: {resized_image.shape}")


