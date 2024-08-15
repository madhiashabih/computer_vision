# from PIL import Image
# import numpy as np
# from applyhomography import applyhomography
# import cv2

# # Claud AI
# def find_homography(pts_src, pts_dst):
#     A = []
#     for i in range(4):
#         x, y = pts_src[i][0], pts_src[i][1]
#         u, v = pts_dst[i][0], pts_dst[i][1]
#         A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
#         A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
#     A = np.array(A)
#     U, S, Vh = np.linalg.svd(A)
#     L = Vh[-1, :] / Vh[-1, -1]
#     H = L.reshape(3, 3)
#     return H

# poster = Image.open('lotr.jpg')
# poster = np.array(poster)
# building = Image.open('griest.jpg')
# building = np.array(building)

# h, w, _ = poster.shape

# pts_dst = np.array([[35, 619], [318,120], [316,561], [107,247] ])
# pts_src = np.array([[0,0], [w-1,0], [w-1,h-1], [0,h-1]])

# H = find_homography(pts_src, pts_dst)
# transformed = applyhomography(poster, H)

# building_pil = Image.fromarray(building)
# transformed_pil = Image.fromarray(transformed.astype('uint8'), 'RGB')

# mask = Image.new('L', transformed_pil.size, 0)
# mask_data = transformed.sum(axis=2) != 0  # True where the image is not black
# mask = Image.fromarray((mask_data * 255).astype('uint8'), 'L')

# building_pil.paste(transformed_pil, (0, 0), mask)
# building_pil.save('output.jpg')
# transformed_pil.save('transform.jpg')

from PIL import Image
import numpy as np

def find_homography(pts_src, pts_dst):
    A = []
    for i in range(4):
        x, y = pts_src[i][0], pts_src[i][1]
        u, v = pts_dst[i][0], pts_dst[i][1]
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)
    return H

def apply_homography(image, H):
    h, w = image.shape[:2]
    y, x = np.indices((h, w))
    homogeneous_coords = np.stack([x.ravel(), y.ravel(), np.ones_like(x).ravel()])
    transformed_coords = H @ homogeneous_coords
    transformed_coords /= transformed_coords[2]
    x_transformed, y_transformed = transformed_coords[:2].round().astype(int)
    
    mask = (x_transformed >= 0) & (x_transformed < w) & (y_transformed >= 0) & (y_transformed < h)
    x_valid, y_valid = x_transformed[mask], y_transformed[mask]
    
    output = np.zeros_like(image)
    output[y_valid, x_valid] = image[y.ravel()[mask], x.ravel()[mask]]
    
    return output

# Load images
poster = Image.open('lotr.jpg')
poster = np.array(poster)
building = Image.open('griest.jpg')
building = np.array(building)

h, w, _ = poster.shape

pts_dst = np.array([[106, 246], [315,124], [35,621], [313,561]])
pts_src = np.array([[0,0], [w-1,0], [0,h-1], [w-1,h-1]])

H = find_homography(pts_src, pts_dst)
transformed = apply_homography(poster, H)

building_pil = Image.fromarray(building)
transformed_pil = Image.fromarray(transformed.astype('uint8'), 'RGB')

mask = Image.new('L', transformed_pil.size, 0)
mask_data = transformed.sum(axis=2) != 0  # True where the image is not black
mask = Image.fromarray((mask_data * 255).astype('uint8'), 'L')

building_pil.paste(transformed_pil, (0, 0), mask)
building_pil.save('output.jpg')
transformed_pil.save('transform.jpg')