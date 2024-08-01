import numpy as np
from PIL import Image
import cv2
import math 
### take a colour image as input and scale (resize) it by a given factor s

### Read in input and store it as an array
def read_image(image):
    img_array = np.array(image)
    return img_array

def save_image(image, path):
    output = Image.fromarray(image)
    output.save(f'{path}')

# Read in image
img = cv2.imread('cat.jpeg')
img = read_image(img)

height, width, channels = img.shape

k = 2
### Initialize scaled image
result = np.zeros((k*height, k*width, channels), dtype=np.uint8)

for i in range(height):
    for j in range(width):
        result[k*i][k*j] = img[i][j] 

# result = k * img 

save_image(result, 'final.jpeg')
