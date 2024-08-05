import numpy as np
from PIL import Image

# Read in greenscreen image
image_path = 'greenscreen.jpg'
image = Image.open(image_path)
img_array = np.array(image)

# Seperate the channels
image = image.convert('RGB')
r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]

mask = g >= 160
mask = mask & (r < 60) 
inverted_mask = ~mask

output = np.zeros_like(img_array)
output[inverted_mask] = img_array[inverted_mask]

background = Image.open('background.png')
background = background.convert('RGB')
background = background.resize(image.size)
background_array = np.array(background)

result_array = np.where(mask[:,:,np.newaxis], background_array, output)

result = Image.fromarray(result_array)
result.save('output.png')