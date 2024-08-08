import cv2
import numpy as np
import sys

def add_salt_and_pepper(image, density):
    noise = np.random.choice([0, 1, 2], size=image.shape[:2], p=[1-density, density/2, density/2])
    image[noise == 1] = 255
    image[noise == 2] = 0
    return image

def median_filter(image, w):
    height, width = image.shape[:2]
    pad = w // 2
    padded_img = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    filtered_img = np.zeros_like(image)
    
    for i in range(height):
        for j in range(width):
            window = padded_img[i:i+w, j:j+w]
            filtered_img[i, j] = np.median(window, axis=(0, 1))
    
    return filtered_img

def process_image(image_path, density):
    img = cv2.imread(image_path)
    if img is None:
    
        raise FileNotFoundError(f"Cannot read image file: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noisy_gray = add_salt_and_pepper(gray, density)
    path = 'output_gray {:.2f}.jpg'.format(density)
    cv2.imwrite(path, noisy_gray)
    
    noisy_color = np.apply_along_axis(lambda x: add_salt_and_pepper(x, density), 2, img)
    path = 'output_color {:.2f}.jpg'.format(density)
    cv2.imwrite(path, noisy_color)
    
    filter = 2
    filtered_img = median_filter(noisy_color, filter)
    path = 'output_filtered {:.2f} {:.2f}.jpg'.format(density, filter)
    cv2.imwrite(path, filtered_img)

if len(sys.argv) != 2:
    print("Usage: python script.py <noise_density>")
    sys.exit(1)
process_image('jean_weight.jpeg', float(sys.argv[1]))
        
