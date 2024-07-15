import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Function to calculate the magnitude of a vector
def magnitude(vector):
    return np.sqrt(np.sum(vector ** 2, axis=1))


# Function to generate random initial positions and velocities for the drones
def generate_initial_conditions():
    drone_a_initial = np.array([np.random.uniform(0, 100), np.random.uniform(0, 3), np.random.uniform(0, 100)])
    drone_b_initial = np.array([np.random.uniform(200, 300), np.random.uniform(0, 3), np.random.uniform(200, 300)])

    target = np.array([100, 80, 400])

    direction_a = target - drone_a_initial

    return drone_a_initial, drone_b_initial, target, direction_a


# Function to check if drones collide
def check_collision(drone_a_initial, velocity_a, drone_b_initial, velocity_b, max_time=10, steps=100):
    t = np.linspace(0, max_time, steps)
    position_a = drone_a_initial + np.outer(t, velocity_a)

    for i in range(steps):
        direction_b = position_a[i] - drone_b_initial
        velocity_b = (direction_b / np.linalg.norm(direction_b)) * np.linalg.norm(
            velocity_b)  # Normalize velocity_b to its speed
        position_b = drone_b_initial + velocity_b * t[i]

        if np.linalg.norm(position_a[i] - position_b) < 1:  # Assuming collision if distance < 1 unit
            return True

    return False


# Function to simulate multiple trials for different speed combinations
def simulate_speed_variations(drone_a_speed, drone_b_speeds, num_trials=100):
    results = np.zeros(len(drone_b_speeds))

    for j, speed_b in enumerate(drone_b_speeds):
        successful_intercepts = 0
        for _ in range(num_trials):
            drone_a_initial, drone_b_initial, target, direction_a = generate_initial_conditions()
            velocity_a = (direction_a / np.linalg.norm(direction_a)) * drone_a_speed

            collision_detected = check_collision(drone_a_initial, velocity_a, drone_b_initial,
                                                 np.array([speed_b, speed_b, speed_b]))
            if collision_detected:
                successful_intercepts += 1

        results[j] = successful_intercepts / num_trials * 100  # Store success rate as percentage

    return results


# Function to plot the results
def plot_results(drone_b_speeds, results, title):
    plt.figure()
    plt.plot(drone_b_speeds, results, marker='o')
    plt.xlabel('Drone B Speed (m/s)')
    plt.ylabel('Success Rate (%)')
    plt.title(title)
    plt.grid(True)
    plt.show()


# Define speed ranges for Drone B and a constant speed for Drone A
drone_a_speed = 40  # Constant speed for Drone A
drone_b_speeds = np.linspace(20, 100, 10)

# Run the simulation multiple times and average the results
num_simulations = 10
aggregated_results = np.zeros(len(drone_b_speeds))

for simulation in range(num_simulations):
    results = simulate_speed_variations(drone_a_speed, drone_b_speeds, num_trials=10000)
    aggregated_results += results

# Average the results
averaged_results = aggregated_results / num_simulations

# Plot the averaged results
plot_results(drone_b_speeds, averaged_results, 'Success Rate of Drone B Intercepting Drone A at Different Speeds')
