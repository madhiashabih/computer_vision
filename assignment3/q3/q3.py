import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image

def dehomogenize(vector):
    return vector[:-1]/vector[-1]

def draw_projected_axes(image_path, P, axis_length=1000):
    img = Image.open(image_path)
    img_array = np.array(img)
    width, height = img.size

    p1, p2, p3, p4 = P[:,0], P[:,1], P[:,2], P[:,3]

    origin = dehomogenize(p4)
    x_vanishing = dehomogenize(p1)
    y_vanishing = dehomogenize(p2)
    z_vanishing = -1*dehomogenize(p3)
    print(x_vanishing)
    print(y_vanishing)
    print(z_vanishing)

    # Create a 2D plot with the image
    plt.figure(figsize=(10, 8))
    plt.imshow(img_array, extent=[0, width, height, 0])  # Display the image with correct extent

    # Plot the lines from the origin to the vanishing points
    plt.plot([origin[0], x_vanishing[0]], [origin[1], x_vanishing[1]], color='r', label='X Axis')
    plt.plot([origin[0], y_vanishing[0]], [origin[1], y_vanishing[1]], color='g', label='Y Axis')
    plt.plot([origin[0], z_vanishing[0]], [origin[1], z_vanishing[1]], color='b', label='Z Axis')

    # Add labels and a legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Projection of Vanishing Points')
    plt.legend()

    # Display the plot
    plt.show()


# Example usage 
if __name__ == "__main__":
    # Replace with your actual camera matrix P and image path
    P = np.array([[ 2.75416038e-03, -2.87589499e-03,  5.85201094e-06,  6.39998608e-01],
                   [-2.34030798e-04, -3.53935397e-04, -3.46138168e-03,  7.68357719e-01],
                   [ 4.74050139e-07,  1.15244558e-07, -5.01027097e-08,  4.22797122e-04]])
    image_path = "lego1.jpg"
    draw_projected_axes(image_path, P)
