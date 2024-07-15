import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Function to calculate the magnitude of a vector
def magnitude(vector):
    return np.sqrt(np.sum(vector ** 2, axis=1))


# Function to generate random initial positions and velocities for the drones
def generate_initial_conditions():
    drone_a_initial = np.array([np.random.uniform(0, 500), np.random.uniform(0, 120), np.random.uniform(0, 500)])
    drone_b_initial = np.array([np.random.uniform(0, 500), np.random.uniform(0, 120), np.random.uniform(0, 500)])

    target = np.array([100, 80, 400])

    direction_a = target - drone_a_initial
    speed_a = 20
    velocity_a = (direction_a / np.linalg.norm(direction_a)) * speed_a

    return drone_a_initial, velocity_a, drone_b_initial


# Function to check if drones collide
def check_collision(drone_a_initial, velocity_a, drone_b_initial, max_time=10, steps=100):
    t = np.linspace(0, max_time, steps)
    position_a = drone_a_initial + np.outer(t, velocity_a)

    for i in range(steps):
        direction_b = position_a[i] - drone_b_initial
        velocity_b = (direction_b / np.linalg.norm(direction_b)) * 23  # Drone B speed is 60 m/s
        position_b = drone_b_initial + velocity_b * t[i]

        if np.linalg.norm(position_a[i] - position_b) < 0.1:  # Assuming collision if distance < 0.1 unit
            return t[i], position_a[i], position_b

    return None, None, None


# Function to simulate multiple trials
def simulate_trials(num_trials=100):
    collision_times = []
    successful_intercepts = 0
    failed_intercepts = 0

    for _ in range(num_trials):
        drone_a_initial, velocity_a, drone_b_initial = generate_initial_conditions()

        collision_time, _, _ = check_collision(drone_a_initial, velocity_a, drone_b_initial)

        if collision_time:
            collision_times.append(collision_time)
            successful_intercepts += 1
        else:
            failed_intercepts += 1

    return collision_times, successful_intercepts, failed_intercepts


# Function to run multiple simulations and compute average success rate
def run_multiple_simulations(num_simulations=10, num_trials=100):
    all_successful_intercepts = 0
    all_failed_intercepts = 0
    all_collision_times = []

    for _ in range(num_simulations):
        collision_times, successful_intercepts, failed_intercepts = simulate_trials(num_trials)
        all_collision_times.extend(collision_times)
        all_successful_intercepts += successful_intercepts
        all_failed_intercepts += failed_intercepts

    overall_success_rate = (all_successful_intercepts / (all_successful_intercepts + all_failed_intercepts)) * 100
    return all_collision_times, all_successful_intercepts, all_failed_intercepts, overall_success_rate


# Function to plot results
def plot_results(collision_times, successful_intercepts, failed_intercepts):
    plt.figure()

    # Plot histogram of collision times
    plt.hist(collision_times, bins=20, alpha=0.7, label='Collision Times')
    plt.axvline(np.mean(collision_times), color='r', linestyle='dashed', linewidth=1, label='Mean Collision Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Collision Times')
    plt.legend()

    # Display successful vs failed intercepts
    labels = ['Successful Intercepts', 'Failed Intercepts']
    counts = [successful_intercepts, failed_intercepts]
    plt.figure()
    plt.bar(labels, counts, color=['green', 'red'])
    plt.title('Successful vs Failed Intercepts')

    plt.show()


# Run the simulations and plot results
num_simulations = 10
num_trials = 1000

collision_times, successful_intercepts, failed_intercepts, overall_success_rate = run_multiple_simulations(num_simulations, num_trials)
plot_results(collision_times, successful_intercepts, failed_intercepts)

print(f"Overall Success Rate: {overall_success_rate:.2f}%")
