import numpy as np
from PIL import Image

def remove_greenscreen():
    # Load images
    fg = np.array(Image.open('greenscreen.jpg').convert('RGB'))
    bg = np.array(Image.open('background.png').convert('RGB').resize(Image.open('greenscreen.jpg').size))

    # Create mask
    mask = (fg[:,:,1] >= 160) & (fg[:,:,0] < 60)

    # Combine foreground and background
    result = np.where(mask[:,:,np.newaxis], bg, fg)

    # Save result
    Image.fromarray(result).save('output.png')
    
     # Save mask
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    mask_image.save('mask.jpg')

remove_greenscreen()