import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_inliers(img1, img2, pts1, pts2, inliers, point_size =4):
    """
    Plot the inlier points between two images.

    Args:
        img1 (np.ndarray): The first image.
        img2 (np.ndarray): The second image.
        pts1 (np.ndarray): Array of (x, y) coordinates of points in the first image.
        pts2 (np.ndarray): Array of (x', y') coordinates of corresponding points in the second image.
        inliers (np.ndarray): The indices of the inlier points.
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the first image
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax1.set_title('Image 1')
    ax1.scatter(pts1[inliers, 0], pts1[inliers, 1], s=point_size, color='r', label='Inliers')
    ax1.legend()

    # Plot the second image
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    ax2.set_title('Image 2')
    ax2.scatter(pts2[inliers, 0], pts2[inliers, 1], s=point_size, color='r', label='Inliers')
    ax2.legend()

    plt.show()
