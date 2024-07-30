import random
import cv2
import sys
import numpy as np

def add_noise(img, density):

    row, col, ch = img.shape
    
    num_pixels = int(density * row * col)
    
    # Salt noise
    salt_coords = [
        (random.randint(0, row - 1), random.randint(0, col - 1))
        for _ in range(num_pixels // 2)
    ]
    img[tuple(zip(*salt_coords))] = 255
    
    # Pepper noise
    pepper_coords = [
        (random.randint(0, row - 1), random.randint(0, col - 1))
        for _ in range(num_pixels // 2)
    ]
    img[tuple(zip(*pepper_coords))] = 0
    
    return img

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <noise_ratio> <input_image_path>")
        sys.exit(1)
    
    try:
        density = float(sys.argv[1])
        if not 0 <= density <= 1:
            raise ValueError("Noise ratio must be between 0 and 1")
    except ValueError as e:
        print(f"Invalid noise ratio: {e}")
        sys.exit(1)
    
    input_path = sys.argv[2]
    
    try:
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read the image: {input_path}")
    except Exception as e:
        print(f"Error reading image: {e}")
        sys.exit(1)
    
    noisy_img = add_noise(img, density)
    
    output_path = 'salt-and-pepper-output.jpg'
    cv2.imwrite(output_path, noisy_img)
    print(f"Noisy image saved as {output_path}")

if __name__ == "__main__":
    main()
