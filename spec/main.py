import numpy as np
import random
import matplotlib.pyplot as plt

# Function to generate random position vector within given ranges
def random_position():
    x = round(random.uniform(0, 500))
    y = round(random.uniform(0, 90))
    z = round(random.uniform(0, 500))
    return np.array([x, y, z], dtype=float)

# Function to generate completely random velocity vector
def random_velocity():
    speed = random.uniform(15, 50)  # Reasonable speed range
    direction = np.random.uniform(-1, 1, 3)  # Random direction in 3D
    direction /= np.linalg.norm(direction)  # Normalize direction
    velocity = direction * speed
    return np.round(velocity)

# Function to find the time at which the drones are at a minimum distance
def time_of_minimum_distance(p1, v1, p2, v2):
    r = p1 - p2
    v = v1 - v2
    t_min = -np.dot(r, v) / np.dot(v, v)
    return max(t_min, 0)

# Generate a scenario
def generate_scenario():
    p1 = random_position()
    p2 = random_position()
    v1 = random_velocity()
    v2 = random_velocity()
    t_min = time_of_minimum_distance(p1, v1, p2, v2)
    pos1_at_t_min = p1 + v1 * t_min
    pos2_at_t_min = p2 + v2 * t_min
    min_distance = np.linalg.norm(pos1_at_t_min - pos2_at_t_min)
    return p1, v1, p2, v2, t_min, min_distance, pos1_at_t_min, pos2_at_t_min

# Generate 1 scenario for visualization
p1, v1, p2, v2, t_min, min_distance, pos1_at_t_min, pos2_at_t_min = generate_scenario()

# Create figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot initial positions and trajectories
ax.plot([p1[0], pos1_at_t_min[0]], [p1[1], pos1_at_t_min[1]], [p1[2], pos1_at_t_min[2]], 'b--', label='Drone 1 Trajectory')
ax.plot([p2[0], pos2_at_t_min[0]], [p2[1], pos2_at_t_min[1]], [p2[2], pos2_at_t_min[2]], 'r--', label='Drone 2 Trajectory')

ax.scatter(p1[0], p1[1], p1[2], color='blue', label='Drone 1 Initial')
ax.scatter(p2[0], p2[1], p2[2], color='red', label='Drone 2 Initial')
ax.scatter(pos1_at_t_min[0], pos1_at_t_min[1], pos1_at_t_min[2], color='cyan', label='Drone 1 at t_min')
ax.scatter(pos2_at_t_min[0], pos2_at_t_min[1], pos2_at_t_min[2], color='magenta', label='Drone 2 at t_min')

# Set labels
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('Drone Trajectories and Minimum Distance Positions')

# Set limits
ax.set_xlim(0, 500)
ax.set_ylim(0, 90)
ax.set_zlim(0, 500)

plt.legend()
plt.show()

print(f"Drone 1 initial position: {p1}, velocity: {v1}")
print(f"Drone 2 initial position: {p2}, velocity: {v2}")
print(f"Time of minimum distance: {t_min:.2f} seconds")
print(f"Minimum distance: {min_distance:.2f} units")
