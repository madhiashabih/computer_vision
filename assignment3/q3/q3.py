# import numpy as np
# import matplotlib.pyplot as plt 
# from PIL import Image

# def dehomogenize(vector):
#     return vector[:-1]/vector[-1]

# def draw_projected_axes(image_path, P, axis_length=1000):
#     img = Image.open(image_path)
#     img_array = np.array(img)
#     width, height = img.size

#     p1, p2, p3, p4 = P[:,0], P[:,1], P[:,2], P[:,3]

#     origin = dehomogenize(p4)
#     x_vanishing = dehomogenize(p1)
#     y_vanishing = dehomogenize(p2)
#     z_vanishing = -1*dehomogenize(p3)
#     print(x_vanishing)
#     print(y_vanishing)
#     print(z_vanishing)

#     # Create a 2D plot with the image
#     plt.figure(figsize=(10, 8))
#     plt.imshow(img_array, extent=[0, width, height, 0])  # Display the image with correct extent

#     # Plot the lines from the origin to the vanishing points
#     plt.plot([origin[0], x_vanishing[0]], [origin[1], x_vanishing[1]], color='r', label='X Axis')
#     plt.plot([origin[0], y_vanishing[0]], [origin[1], y_vanishing[1]], color='g', label='Y Axis')
#     plt.plot([origin[0], z_vanishing[0]], [origin[1], z_vanishing[1]], color='b', label='Z Axis')

#     # Add labels and a legend
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('2D Projection of Vanishing Points')
#     plt.legend()

#     # Display the plot
#     plt.show()


# # Example usage 
# if __name__ == "__main__":
#     # Replace with your actual camera matrix P and image path
#     P = np.array([[ 2.75416038e-03, -2.87589499e-03,  5.85201094e-06,  6.39998608e-01],
#                    [-2.34030798e-04, -3.53935397e-04, -3.46138168e-03,  7.68357719e-01],
#                    [ 4.74050139e-07,  1.15244558e-07, -5.01027097e-08,  4.22797122e-04]])
#     image_path = "lego1.jpg"
#     draw_projected_axes(image_path, P)

#     P = np.array([
#     [-3.45737581e-03,  1.84680728e-03, -1.34003533e-05, -4.87876478e-01],
#     [ 2.22192346e-04,  3.96427196e-04,  3.84370894e-03, -8.72895156e-01],
#     [-2.06695621e-07, -3.04302600e-07,  4.29616113e-08, -4.94925813e-04]
#     ])

#     image_path = "lego2.jpg"
#     draw_projected_axes(image_path, P)

import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
from typing import Tuple

def dehomogenize(vector: np.ndarray) -> np.ndarray:
    """Dehomogenize a vector by dividing by its last component."""
    return vector[:-1] / vector[-1]

def compute_vanishing_points(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute origin and vanishing points from the camera matrix."""
    origin = dehomogenize(P[:, 3])
    x_vanishing = dehomogenize(P[:, 0])
    y_vanishing = dehomogenize(P[:, 1])
    z_vanishing = -dehomogenize(P[:, 2])  # Negative for correct orientation
    return origin, x_vanishing, y_vanishing, z_vanishing

def draw_projected_axes(image_path: str, P: np.ndarray, axis_length: float = 1000):
    """Draw projected axes on an image given the camera matrix."""
    # Load and process image
    img = Image.open(image_path)
    img_array = np.array(img)
    width, height = img.size

    # Compute vanishing points
    origin, x_vanishing, y_vanishing, z_vanishing = compute_vanishing_points(P)

    # Print vanishing points for debugging
    print(f"X vanishing point: {x_vanishing}")
    print(f"Y vanishing point: {y_vanishing}")
    print(f"Z vanishing point: {z_vanishing}")

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img_array, extent=[0, width, height, 0])

    # Plot axes
    axes = [
        (x_vanishing, 'r', 'X Axis'),
        (y_vanishing, 'g', 'Y Axis'),
        (z_vanishing, 'b', 'Z Axis')
    ]
    
    for vanishing_point, color, label in axes:
        ax.plot([origin[0], vanishing_point[0]], [origin[1], vanishing_point[1]], 
                color=color, linewidth=2, label=label)

    # Customize plot
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Invert y-axis to match image coordinates
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Projection of Vanishing Points')
    ax.legend()

    plt.tight_layout()
    plt.show()

def visualize_multiple_projections(projection_data: list):
    """Visualize projections for multiple camera matrices and images."""
    for i, (P, image_path) in enumerate(projection_data, 1):
        print(f"\nProcessing projection {i}:")
        draw_projected_axes(image_path, P)

if __name__ == "__main__":
    projection_data = [
        (np.array([
            [ 2.75416038e-03, -2.87589499e-03,  5.85201094e-06,  6.39998608e-01],
            [-2.34030798e-04, -3.53935397e-04, -3.46138168e-03,  7.68357719e-01],
            [ 4.74050139e-07,  1.15244558e-07, -5.01027097e-08,  4.22797122e-04]
        ]), "lego1.jpg"),
        (np.array([
            [-3.45737581e-03,  1.84680728e-03, -1.34003533e-05, -4.87876478e-01],
            [ 2.22192346e-04,  3.96427196e-04,  3.84370894e-03, -8.72895156e-01],
            [-2.06695621e-07, -3.04302600e-07,  4.29616113e-08, -4.94925813e-04]
        ]), "lego2.jpg")
    ]

    visualize_multiple_projections(projection_data)