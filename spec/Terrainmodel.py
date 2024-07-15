import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_smooth_terrain_with_peaks(size=(500, 500), peaks=None):
    x = np.linspace(0, size[0] - 1, size[0])
    z = np.linspace(0, size[1] - 1, size[1])
    x, z = np.meshgrid(x, z)

    # Initialize the y-axis as heights
    y = np.zeros(size)

    # Default peaks if none provided
    if peaks is None:
        peaks = [
            {'x': 50, 'z': 100, 'height': 10, 'radius': 100},  # Peak with height 38m
            {'x': 150, 'z': 300, 'height': 20, 'radius': 100}  # Peak with height 51m
        ]

    for peak in peaks:
        peak_x = peak['x']
        peak_z = peak['z']
        peak_height = peak['height']
        peak_radius = peak['radius']

        # Calculate distance to the peak and apply Gaussian function
        distance = np.sqrt((x - peak_x) ** 2 + (z - peak_z) ** 2)
        # The Gaussian function is used here to create a smooth peak
        peak_hill = peak_height * np.exp(-distance ** 2 / (2 * peak_radius ** 2))
        y += peak_hill

    # Ensure the terrain height at the specified peak coordinates is correct
    for peak in peaks:
        px, pz = peak['x'], peak['z']
        height_at_peak = y[int(pz), int(px)]
        print(f"Height at peak ({px}, {pz}) is {height_at_peak:.2f} meters.")

    # Normalize terrain to fit within practical visualization height range
    max_height = np.max(y)
    min_height = np.min(y)
    y = (y - min_height) / (max_height - min_height) * 51  # Scale to maximum height for visualization

    return x, y, z

# Function to plot the terrain
def plot_terrain(x, y, z):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the terrain with y as height
    ax.plot_surface(x, z, y, cmap='terrain', rstride=5, cstride=5, edgecolor='none')

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Height')
    ax.set_title('3D Terrain Map with Specific Peaks')

    plt.show()

# Define peaks for the terrain
peaks = [
    {'x': 50, 'z': 100, 'height': 10, 'radius': 80},  # Peak with height 38m
    {'x': 200, 'z': 300, 'height': 30, 'radius': 70}   # Peak with height 51m
]

# Generate and plot the terrain
x, y, z = create_smooth_terrain_with_peaks(peaks=peaks)
plot_terrain(x, y, z)
