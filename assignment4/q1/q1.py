import cv2
import matplotlib.pyplot as plt 
import numpy as np
import random
plt.rcParams['figure.figsize'] = [15, 15]
from applyhomography import applyhomography
from PIL import Image  

def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def filter_matches(matches, max_distance):
    return [match for match in matches if calculate_distance(*match) <= max_distance]

def plot_matches(src_img, matches, max_distance):
    # Convert the source image to grayscale
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    
    filtered_matches = filter_matches(matches, max_distance)
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Display the grayscale image
    ax.imshow(gray_img, cmap='gray')
    
    # Plot the matches
    for match in filtered_matches:
        x1, y1, x2, y2 = match
        ax.plot([x1, x2], [y1, y2], 'r-', linewidth=0.5)
        ax.plot(x1, y1, 'bo', markersize=5)  # Blue circle for xi
    
    ax.set_title(f'Feature Matches on Grayscale Image (Max Distance: {max_distance:.2f})')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# Read the data from the text file
data = []
with open('ET/matches.txt', 'r') as file:
    for line in file:
        row = [float(x) for x in line.strip().split()]
        data.append(row)

# Extract the (x, y) and (x', y') coordinates
src_pts = np.array([[x, y] for x, y, x_, y_ in data], dtype=np.float32)
dst_pts = np.array([[x_, y_] for x, y, x_, y_ in data], dtype=np.float32)

# Load the source and destination images
src_img = cv2.imread('ET/et1.jpg', cv2.IMREAD_COLOR)

matches = np.hstack((src_pts, dst_pts))

# Create a combined image with the source and destination images side-by-side
max_distance = 10
plot_matches(src_img, matches, max_distance)
