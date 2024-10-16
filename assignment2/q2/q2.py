import numpy as np
import numpy.linalg as la
from PIL import Image

def apply_homography(A, H):
    A = A.astype(float)
    m, n, c = A.shape
    
    # Forward transform the corners
    corners = np.array([[0, 0, 1], [n-1, 0, 1], [0, m-1, 1], [n-1, m-1, 1]]).T
    transformed_corners = np.dot(H, corners)
    transformed_corners /= transformed_corners[2, :]
    
    minx = np.floor(np.min(transformed_corners[0, :]))
    maxx = np.ceil(np.max(transformed_corners[0, :]))
    miny = np.floor(np.min(transformed_corners[1, :]))
    maxy = np.ceil(np.max(transformed_corners[1, :]))
    
    nn = int(maxx - minx)
    mm = int(maxy - miny)
    
    B = np.zeros((mm, nn, c)) + 255
    Hi = la.inv(H)
    
    for x in range(nn):
        for y in range(mm):
            p = np.array([x + minx, y + miny, 1]).reshape((3, 1))
            pp = np.dot(Hi, p)
            xp, yp = pp[0]/pp[2], pp[1]/pp[2]
            
            xpf, xpc = int(np.floor(xp)), int(np.floor(xp)) + 1
            ypf, ypc = int(np.floor(yp)), int(np.floor(yp)) + 1
            
            if 0 <= xpf < n and 0 <= ypf < m and 0 <= xpc < n and 0 <= ypc < m:
                B[y, x, :] = (
                    (xpc - xp) * (ypc - yp) * A[ypf, xpf, :] +
                    (xpc - xp) * (yp - ypf) * A[ypc, xpf, :] +
                    (xp - xpf) * (ypc - yp) * A[ypf, xpc, :] +
                    (xp - xpf) * (yp - ypf) * A[ypc, xpc, :]
                )
            else:
                B[y, x, :] = 255  # White pixel where no mapping occurs
    
    return B.astype(np.uint8)

def find_homography(pts_src, pts_dst):
    A = []
    for i in range(4):
        x, y = pts_src[i][0], pts_src[i][1]
        u, v = pts_dst[i][0], pts_dst[i][1]
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    A = np.array(A)
    print(A)
    _, _, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    return L.reshape(3, 3)

# Load images
poster = Image.open('amuse.jpg')
building = np.array(Image.open('griest.jpg'))

# Resize image
# n_w = 250
# n_h = 350
# resized_poster = poster.resize((n_w, n_h), Image.Resampling.LANCZOS)

poster = np.array(poster)

h, w, _ = poster.shape

pts_dst = np.array([[106, 246], [315, 124], [35, 621], [313, 561]])
pts_src = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])

H = find_homography(pts_src, pts_dst)
print(H)
transformed = apply_homography(poster, H)

# Compensate the paste operation using the offset
offset_x, offset_y = int(min(pts_dst[:, 0])), int(min(pts_dst[:, 1]))

building_pil = Image.fromarray(building)
transformed_pil = Image.fromarray(transformed)

# Improved mask creation: check if any pixel in any channel is different from 255 (white)
mask_data = np.any(transformed != 255, axis=2)
mask = Image.fromarray((mask_data * 255).astype('uint8'), 'L')

# Paste the transformed image onto the building image
building_pil.paste(transformed_pil, (offset_x, offset_y), mask)
building_pil.save('output.jpg')
transformed_pil.save('transform.jpg')

