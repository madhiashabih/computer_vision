import numpy as np
from PIL import Image
import cv2

### Read in input and store it as an array
def read_image(image):
    img_array = np.array(image)
    return img_array

def save_image(image, path):
    output = Image.fromarray(image)
    output.save(f'{path}')

img = cv2.imread('fruits.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

### Grayscale
gr_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurImg = cv2.blur(gr_img,(2,2))

img = read_image(gr_img)
blurImg = read_image(blurImg)
save_image(blurImg, 'blur.jpeg')

mask = img - blurImg

save_image(mask, 'mask.jpeg')

unsharp = img + mask
save_image(unsharp, 'output.jpeg')
