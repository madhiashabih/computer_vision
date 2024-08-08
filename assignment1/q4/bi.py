import numpy as np
import cv2

def bl_resize(original_img, new_h, new_w):
    old_h, old_w, c = original_img.shape
    resized = np.zeros((new_h, new_w, c), dtype=np.uint8)
    w_scale = old_w / new_w
    h_scale = old_h / new_h

    for i in range(new_h):
        for j in range(new_w):
            x, y = i * h_scale, j * w_scale
            x0, x1 = int(np.floor(x)), min(old_h - 1, int(np.ceil(x)))
            y0, y1 = int(np.floor(y)), min(old_w - 1, int(np.ceil(y)))

            if x0 == x1 and y0 == y1:
                q = original_img[x0, y0]
            elif x0 == x1:
                q = original_img[x0, y0] * (y1 - y) + original_img[x0, y1] * (y - y0)
            elif y0 == y1:
                q = original_img[x0, y0] * (x1 - x) + original_img[x1, y0] * (x - x0)
            else:
                q1 = original_img[x0, y0] * (x1 - x) + original_img[x1, y0] * (x - x0)
                q2 = original_img[x0, y1] * (x1 - x) + original_img[x1, y1] * (x - x0)
                q = q1 * (y1 - y) + q2 * (y - y0)

            resized[i, j] = q

    return resized

img = cv2.imread('fruits.jpeg')
k = 3
output = bl_resize(img, k * img.shape[0], k * img.shape[1])
cv2.imwrite(f'output_bi_{k:.2f}.jpeg', output)
