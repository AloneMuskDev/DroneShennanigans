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
        position_b = drone_b_initial + np.outer(t[:i + 1], velocity_b)[-1]

        if np.linalg.norm(position_a[i] - position_b) < 1:  # Assuming collision if distance < 1 unit
            return True

    return False


# Function to simulate multiple trials for different speed combinations
def simulate_speed_variations(drone_a_speeds, drone_b_speeds, num_trials=100):
    results = np.zeros((len(drone_a_speeds), len(drone_b_speeds)))

    for i, speed_a in enumerate(drone_a_speeds):
        for j, speed_b in enumerate(drone_b_speeds):
            successful_intercepts = 0
            for _ in range(num_trials):
                drone_a_initial, drone_b_initial, target, direction_a = generate_initial_conditions()
                velocity_a = (direction_a / np.linalg.norm(direction_a)) * speed_a
                velocity_b = np.array([speed_b, speed_b, speed_b])  # Constant speed for Drone B

                collision_detected = check_collision(drone_a_initial, velocity_a, drone_b_initial, velocity_b)
                if collision_detected:
                    successful_intercepts += 1

            results[i, j] = successful_intercepts / num_trials * 100  # Store success rate as percentage

    return results


# Function to plot the results
def plot_results(drone_a_speeds, drone_b_speeds, results, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(drone_b_speeds, drone_a_speeds)
    ax.plot_surface(X, Y, results, cmap='viridis')

    ax.set_xlabel('Drone B Speed (m/s)')
    ax.set_ylabel('Drone A Speed (m/s)')
    ax.set_zlabel('Success Rate (%)')
    ax.set_title(title)

    plt.show()


# Define speed ranges for Drone A and Drone B
drone_a_speeds = np.linspace(10, 60, 6)
drone_b_speeds = np.linspace(20, 100, 6)

# Run the simulation multiple times and average the results
num_simulations = 100
aggregated_results = np.zeros((len(drone_a_speeds), len(drone_b_speeds)))

for simulation in range(num_simulations):
    results = simulate_speed_variations(drone_a_speeds, drone_b_speeds, num_trials=10)
    aggregated_results += results

# Average the results
averaged_results = aggregated_results / num_simulations

# Plot the averaged results
plot_results(drone_a_speeds, drone_b_speeds, averaged_results, 'Averaged Success Rate of Drone B Intercepting Drone A')
