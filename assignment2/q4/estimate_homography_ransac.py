import numpy as np
import cv2

def estimate_homography_ransac(pts1, pts2, reproj_threshold=4.0, num_iterations=500):
    """
    Estimate a homography matrix using RANSAC.

    Args:
        pts1 (np.ndarray): Array of (x, y) coordinates of points in the first image.
        pts2 (np.ndarray): Array of (x', y') coordinates of corresponding points in the second image.
        reproj_threshold (float): Maximum allowed reprojection error to consider a point an inlier.
        num_iterations (int): Number of RANSAC iterations to perform.

    Returns:
        np.ndarray: The estimated homography matrix.
        np.ndarray: The indices of the inlier points.
    """
    best_H = None
    best_inliers = None
    best_num_inliers = 0

    for i in range(num_iterations):
        # Randomly select 4 matches
        indices = np.random.choice(len(pts1), size=4, replace=False)
        p1 = pts1[indices]
        p2 = pts2[indices]

        # Compute the homography matrix from the 4 matches
        H, _ = cv2.findHomography(p1, p2, method=cv2.RANSAC)

        # Evaluate the homography matrix on all matches
        pts1_proj = cv2.perspectiveTransform(pts1.reshape(-1, 1, 2), H)
        pts1_proj = pts1_proj.reshape(-1, 2)
        distances = np.linalg.norm(pts2 - pts1_proj, axis=1)

        # Count the number of inliers
        num_inliers = np.sum(distances < reproj_threshold)

        # Update the best homography and inliers if necessary
        if num_inliers > best_num_inliers:
            best_H = H
            best_inliers = np.where(distances < reproj_threshold)[0]
            best_num_inliers = num_inliers

    # Refine the homography matrix using the inliers
    refined_H, _ = cv2.findHomography(pts1[best_inliers], pts2[best_inliers], method=cv2.LMEDS)

    return refined_H, best_inliers
