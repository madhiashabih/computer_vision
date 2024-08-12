import numpy as np
from applyhomography import applyhomography
import math
from PIL import Image

# Transforms a colour image of your choice with a few different similarities 
# Equation: x = Hx,

# https://ryansblog2718281.medium.com/image-processing-projective-transformation-c6795af1c11

image = Image.open('afghanGirl.jpg')
image = np.array(image) 

s_vals = [0.5, 1, 2]
tx_vals = [0.5, 1, 2]
ty_vals = [0.5, 1, 2]

for s in s_vals:
    for tx in tx_vals:
        for ty in ty_vals:
            similarities = np.array([
                [s*math.cos(math.pi/4), -s*math.sin(math.pi/4), tx],
                [s*math.sin(math.pi/4), s*math.cos(math.pi/4),  ty],
                [0, 0, 1]
            ])

        transformed = applyhomography(image, similarities)

        output = Image.fromarray(transformed.astype('uint8'))
        output.save(f'output/similar_{s}_{tx}_{ty}.jpg')


