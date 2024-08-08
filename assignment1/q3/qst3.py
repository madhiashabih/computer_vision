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

blur_size=(2, 2) 
strength=1.0
   
# Read image
img = read_image('cat.jpeg')

# Apply unsharp mask
sharpened, mask = apply_unsharp_mask(img, blur_size, strength)

# Save results
save_image(sharpened, 'output.jpg')
save_image(mask, 'mask.jpg')

