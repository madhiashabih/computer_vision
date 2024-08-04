import numpy as np
from PIL import Image
import cv2
import math 
import matplotlib.pyplot as plt

### Take a colour image as input and scale (resize) it by a given factor s. ###

# Read in image
img = cv2.imread('fruits.jpeg', cv2.IMREAD_COLOR)

k = 2

height, width, channels = img.shape

# Set the size of the new image
col = img.shape[1] * k
row = img.shape[0] * k 

# Initialize scaled image
result = np.zeros((k*height, k*width, channels), dtype=np.uint8)

# Nearest Neighbour Interpolation
for i in range(height):
    for j in range(width):
        result[k*i][k*j] = img[i][j]
        result[(k*i)-1][k*j] = img[i][j]
        result[k*i][(k*j)-1] = img[i][j]
        result[(k*i)-1][(k*j)-1] = img[i][j]

cv2.imwrite('output_fruit.jpeg', result)




