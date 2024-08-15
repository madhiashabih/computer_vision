from PIL import Image
import numpy as np
from applyhomography import applyhomography
import cv2

# Claud AI
def find_homography(src_points, dst_points):
    A = []
    for i in range(4):
        x, y = src_points[i][0], src_points[i][1]
        u, v = dst_points[i][0], dst_points[i][1]
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)
    return H

poster = Image.open('lotr.jpg')
poster = np.array(poster)
building = Image.open('griest.jpg')
building = np.array(building)

h, w, _ = poster.shape

pts_dst = np.array([[35, 619], [107,247],[318,120],[316,561]])
pts_src = np.array([[0,0], [w-1,0], [w-1,h-1], [0,h-1]])

# Apply a projective
H = find_homography(pts_src, pts_dst)
transformed = applyhomography(poster, H)

# Convert building to PIL Image
building_pil = Image.fromarray(building)

# Convert transformed poster to PIL Image
transformed_pil = Image.fromarray(transformed.astype('uint8'), 'RGB')

# Create a mask for the transformed poster
mask = Image.new('L', transformed_pil.size, 0)
mask_data = transformed.sum(axis=2) != 0  # True where the image is not black
mask = Image.fromarray((mask_data * 255).astype('uint8'), 'L')

# Paste the transformed poster onto the building
building_pil.paste(transformed_pil, (0, 0), mask)

# Save the final output
building_pil.save('output.jpg')

# If you want to save the transformed poster separately
transformed_pil.save('transform.jpg')

