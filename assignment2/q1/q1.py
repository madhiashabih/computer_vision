import numpy as np
from applyhomography import applyhomography
import math
from PIL import Image

# Transforms a colour image of your choice with a few different similarities 
# Equation: x = Hx,

# https://ryansblog2718281.medium.com/image-processing-projective-transformation-c6795af1c11

image = Image.open('afghanGirl.jpg')
image = np.array(image) 

s_vals = [0.5, 0, 1, 2]
tx = 1
ty = 1
div = [3,4,6]

for s in s_vals:
    for d in div:
        print(f"Processing s={s}, d={d}")
        similarities = np.array([
            [s*math.cos(math.pi/d), -s*math.sin(math.pi/d), tx],
            [s*math.sin(math.pi/d), s*math.cos(math.pi/d),  ty],
            [0, 0, 1]
        ])

        try:
            transformed = applyhomography(image, similarities)
            output = Image.fromarray(transformed.astype('uint8'))
            output.save(f'output/similar_d:_{d}_s:_{s}_tx:_{tx}_ty_{ty}.jpg')
        except Exception as e:
            print(f"Error processing s={s}, d={d}: {e}")

