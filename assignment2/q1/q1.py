import numpy as np
from applyhomography import applyhomography
import math
from PIL import Image
import cv2 
# Transforms a colour image of your choice with a few different similarities 
# Equation: x = Hx,

# https://ryansblog2718281.medium.com/image-processing-projective-transformation-c6795af1c11

image = Image.open('afghanGirl.jpg')
image = np.array(image) 

scale = np.array([
    [2, 0, 0], 
    [0, 2, 0],
    [0, 0, 1]
])

translation = np.array([
    [1, 0, 2],
    [0, 1, 2],
    [0, 0, 1]
])
rotation = ([
    [math.cos(math.pi/4), -math.sin(math.pi/4), 0],
    [math.sin(math.pi/4), math.cos(math.pi/4),  0],
    [0, 0, 1]
])

scaled = applyhomography(image, scale)
translated = applyhomography(image, translation)
rotation = applyhomography(image, rotation)

output = Image.fromarray(scaled)
output.save('scaled.jpg')

output = Image.fromarray(translated)
output.save('translated.jpg')

output = Image.fromarray(rotation)
output.save('rotation.jpg')
