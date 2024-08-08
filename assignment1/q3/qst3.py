import numpy as np
import cv2
from PIL import Image

def read_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image file: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_image(image, path):
    Image.fromarray(image.astype(np.uint8)).save(path)

def normalize_mask(mask):
    # Shift and scale mask to [0, 255] range
    mask_min, mask_max = mask.min(), mask.max()
    return ((mask - mask_min) / (mask_max - mask_min) * 255).astype(np.uint8)

def apply_unsharp_mask(image, blur_size=(2, 2), strength=1.0):
    # Convert to grayscale if the image is color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Apply blur
    blurred = cv2.blur(gray, blur_size)

    # Create mask
    mask = gray.astype(float) - blurred.astype(float)

    # Normalize mask for visualization
    normalized_mask = normalize_mask(mask)

    # Apply unsharp mask
    sharpened = gray + strength * mask

    # Clip values to valid range
    return np.clip(sharpened, 0, 255).astype(np.uint8), normalized_mask

def main(input_path, output_path, mask_path, blur_size=(2, 2), strength=1.0):
    try:
        # Read image
        img = read_image(input_path)

        # Apply unsharp mask
        sharpened, mask = apply_unsharp_mask(img, blur_size, strength)

        # Save results
        save_image(sharpened, output_path)
        save_image(mask, mask_path)

        print(f"Sharpened image saved as {output_path}")
        print(f"Mask image saved as {mask_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Apply unsharp mask to an image")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("output", help="Path for output sharpened image")
    parser.add_argument("mask", help="Path for output mask image")
    parser.add_argument("--blur", nargs=2, type=int, default=[2, 2], help="Blur kernel size (default: 2 2)")
    parser.add_argument("--strength", type=float, default=1.0, help="Strength of sharpening effect (default: 1.0)")
    args = parser.parse_args()

    main(args.input, args.output, args.mask, tuple(args.blur), args.strength)