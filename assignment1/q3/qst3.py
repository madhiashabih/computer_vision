import numpy as np
from PIL import Image
import cv2

### Read in input and store it as an array
def read_image(image):
    img_array = np.array(image)
    return img_array

def save_image(image, path):
    output = Image.fromarray(image)
    output.save(f'assignment1/q3/{path}')
    
img = cv2.imread('assignment1/q3/fruits.jpeg')
blurImg = cv2.blur(img,(10,10))

img = read_image(img)
print(img)
blurImg = read_image(blurImg)
print(blurImg)
save_image(blurImg, 'blur.jpeg')

mask = img - blurImg

save_image(mask, 'mask.jpeg')

unsharp = blurImg + mask
save_image(unsharp, 'final.jpeg')