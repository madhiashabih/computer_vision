import cv2
import numpy as np
import sys

def median_filter(image, w):
    # Ensure w is odd
    if w % 2 == 0:
        w += 1
    
    # Pad the image
    pad = w // 2
    padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    
    # Create output image
    output = np.zeros_like(image)
    
    # Apply median filter to each channel
    for c in range(image.shape[2]):  # Iterate over channels
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                neighborhood = padded_image[i:i+w, j:j+w, c]
                output[i, j, c] = np.median(neighborhood)
    
    return output

def add_salt_and_pepper(image, density):
    noisy = np.copy(image)
    num_salt = np.ceil(density * image.size * 0.5).astype(int)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy[tuple(coords)] = 255
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy[tuple(coords)] = 0
    return noisy

def process_image(image_path, density):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image file: {image_path}")
    
    # Process grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noisy_gray = add_salt_and_pepper(gray, density)
    cv2.imwrite('output_gray.jpg', noisy_gray)
    
    # Process color
    noisy_color = cv2.merge([add_salt_and_pepper(channel, density) for channel in cv2.split(img)])
    cv2.imwrite('output_color.jpg', noisy_color)
    
    # Apply median filter
    filtered_img = median_filter(noisy_color, 3)
    
    # Save the result
    cv2.imwrite('median_filtered.jpg', filtered_img)

    

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <image_path> <noise_density>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    try:
        density = float(sys.argv[2])
        if not 0 <= density <= 1:
            raise ValueError("Noise density must be between 0 and 1")
    except ValueError as e:
        print(f"Invalid noise density: {e}")
        sys.exit(1)
    
    try:
        process_image(image_path, density)
        print("Processing complete. Check output_gray.jpg and output_color.jpg")
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    main()