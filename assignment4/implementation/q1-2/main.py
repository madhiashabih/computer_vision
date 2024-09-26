import matplotlib.pyplot as plt 
import numpy as np


# Import functions from other modules (assuming they exist)
from functions import draw_plot_3d, load_data, process_matches, compute_fundamental_matrix, compute_essential_matrix, compute_projection_matrices, find_valid_projection_matrix, compute_rectification_transforms, apply_rectification, save_images
from applyhomography import applyhomography

plt.rcParams['figure.figsize'] = [15, 15]

def main():
    # Load data
    matches, src_img, dst_img = load_data('ET/matches.txt', 'ET/et1.jpg', 'ET/et2.jpg')
    
    # Process matches
    filtered_matches = process_matches(matches, src_img)
    
    # Compute fundamental matrix
    inliers_1, inliers_2, F = compute_fundamental_matrix(src_img, filtered_matches)
    print("\nFundamental matrix F:")
    print(F)
    
    # Load camera matrix K
    K = np.loadtxt('ET/K.txt')
    print("\nCamera matrix K:")
    print(K)
    
    # Compute essential matrix
    E, U, S, Vt = compute_essential_matrix(F, K)
    print("\nEssential matrix E:")
    print(E)
    
    print("\nMatrix U:")
    print(U)
    print("\nSingular values (S):")
    print(S)
    print("\nMatrix V^T:")
    print(Vt)
    
    # Compute projection matrices
    P, potential_P_primes = compute_projection_matrices(K, U, Vt)
    print("\nProjection matrix P:")
    print(P)
    
    # Find valid projection matrix
    P_prime, R_prime = find_valid_projection_matrix(P, potential_P_primes, inliers_1, inliers_2)
    print("\nP-prime:")
    print(P_prime)
    
    # Plot 3D
    draw_plot_3d(np.array(inliers_1), np.array(inliers_2), P, P_prime, src_img, dst_img, "output/et_3d")
    
    # Compute rectification transforms
    T1, T2 = compute_rectification_transforms(P, P_prime)
    
    # Apply rectification
    transformed_1, transformed_2 = apply_rectification(src_img, dst_img, T1, T2)
    
    # Save transformed images
    save_images(transformed_1, transformed_2, "output/et1_t.jpg", "output/et2_t.jpg")

if __name__ == "__main__":
    main()