import cv2
import numpy as np
import sys

def add_salt_and_pepper(image, density):
    noise = np.random.choice([0, 1, 2], size=image.shape[:2], p=[1-density, density/2, density/2])
    image[noise == 1] = 255
    image[noise == 2] = 0
    return image

def median_filter(image, w):
    return cv2.medianBlur(image, w)

def process_image(image_path, density):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image file: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noisy_gray = add_salt_and_pepper(gray, density)
    cv2.imwrite('output_gray.jpg', noisy_gray)
    
    noisy_color = np.apply_along_axis(lambda x: add_salt_and_pepper(x, density), 2, img)
    cv2.imwrite('output_color.jpg', noisy_color)
    
    filtered_img = median_filter(noisy_color, 3)
    cv2.imwrite('median_filtered.jpg', filtered_img)

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <image_path> <noise_density>")
        sys.exit(1)
    
    try:
        process_image(sys.argv[1], float(sys.argv[2]))
        print("Processing complete. Check output files.")
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()