import numpy as np
from PIL import Image

image_path = 'material/greenscreen.jpg'
image = Image.open(image_path)
img_array = np.array(image)

image = image.convert('RGB')

r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]

mask = (g >= 160) & (g <= 255)

mask = mask & (g > r) & (g > b) 

inverted_mask = ~mask

output = np.zeros_like(img_array)
output[inverted_mask] = img_array[inverted_mask]

result = Image.fromarray(output)
result.save('output.png')