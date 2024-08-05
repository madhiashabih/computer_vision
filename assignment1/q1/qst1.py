import numpy as np
from PIL import Image

def remove_greenscreen(fg_path, bg_path, output_path):
    # Load images
    fg = np.array(Image.open(fg_path).convert('RGB'))
    bg = np.array(Image.open(bg_path).convert('RGB').resize(Image.open(fg_path).size))

    # Create mask
    mask = (fg[:,:,1] >= 160) & (fg[:,:,0] < 60)

    # Combine foreground and background
    result = np.where(mask[:,:,np.newaxis], bg, fg)

    # Save result
    Image.fromarray(result).save(output_path)

remove_greenscreen('greenscreen.jpg', 'background.png', 'output.png')