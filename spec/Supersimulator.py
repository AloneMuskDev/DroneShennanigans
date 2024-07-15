import numpy as np
import matplotlib.pyplot as plt


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
            return True, t[i]

    return False, None


# Function to simulate multiple trials for different speed combinations
def simulate_speed_variations(drone_a_speeds, drone_b_speed, num_trials=100):
    results = np.zeros(len(drone_a_speeds))

    for i, speed_a in enumerate(drone_a_speeds):
        successful_intercepts = 0
        for _ in range(num_trials):
            drone_a_initial, drone_b_initial, target, direction_a = generate_initial_conditions()
            velocity_a = (direction_a / np.linalg.norm(direction_a)) * speed_a

            collision_detected, _ = check_collision(drone_a_initial, velocity_a, drone_b_initial,
                                                    np.array([drone_b_speed, drone_b_speed, drone_b_speed]))
            if collision_detected:
                successful_intercepts += 1

        results[i] = successful_intercepts / num_trials * 100  # Store success rate as percentage

    return results


# Function to analyze the effect of initial separation distance on success rate
def simulate_initial_separation_distance_variations(drone_a_speed, drone_b_speed, separation_distances, num_trials=10000):
    results = np.zeros(len(separation_distances))
    for i, sep_dist in enumerate(separation_distances):
        successful_intercepts = 0
        for _ in range(num_trials):
            drone_a_initial = np.array([0, 1, 0])
            drone_b_initial = np.array([sep_dist, 1, sep_dist])

            target = np.array([100, 80, 400])
            direction_a = target - drone_a_initial
            velocity_a = (direction_a / np.linalg.norm(direction_a)) * drone_a_speed

            collision_detected, _ = check_collision(drone_a_initial, velocity_a, drone_b_initial,
                                                    np.array([drone_b_speed, drone_b_speed, drone_b_speed]))
            if collision_detected:
                successful_intercepts += 1

        results[i] = successful_intercepts / num_trials * 100  # Store success rate as percentage

    return results


# Function to simulate the impact of Drone B's speed
def simulate_drone_b_speed_variations(drone_a_speed, drone_b_speeds, num_trials=100):
    results = np.zeros(len(drone_b_speeds))

    for i, speed_b in enumerate(drone_b_speeds):
        successful_intercepts = 0
        for _ in range(num_trials):
            drone_a_initial, drone_b_initial, target, direction_a = generate_initial_conditions()
            velocity_a = (direction_a / np.linalg.norm(direction_a)) * drone_a_speed

            collision_detected, _ = check_collision(drone_a_initial, velocity_a, drone_b_initial,
                                                    np.array([speed_b, speed_b, speed_b]))
            if collision_detected:
                successful_intercepts += 1

        results[i] = successful_intercepts / num_trials * 100  # Store success rate as percentage

    return results


# Function to plot the results
def plot_results(x_values, y_values, x_label, y_label, title):
    plt.figure()
    plt.plot(x_values, y_values, marker='o')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.show()


# Define ranges and parameters for analysis
drone_a_speeds = np.linspace(10, 50, 10)
drone_b_speeds = np.linspace(20, 60, 10)
separation_distances = np.linspace(100, 300, 6)

# Run simulations and plot results
# Success Rate vs. Drone A Speed
results_drone_a_speed = simulate_speed_variations(drone_a_speeds, drone_b_speed=40, num_trials=100)
plot_results(drone_a_speeds, results_drone_a_speed, 'Drone A Speed (m/s)', 'Success Rate (%)',
             'Success Rate vs. Drone A Speed')

# Success Rate vs. Initial Separation Distance
results_initial_separation_distance = simulate_initial_separation_distance_variations(drone_a_speed=30,
                                                                                      drone_b_speed=40,
                                                                                      separation_distances=separation_distances,
                                                                                      num_trials=100)
plot_results(separation_distances, results_initial_separation_distance, 'Initial Separation Distance (m)',
             'Success Rate (%)', 'Success Rate vs. Initial Separation Distance')

# Success Rate vs. Drone B Speed
results_drone_b_speed = simulate_drone_b_speed_variations(drone_a_speed=30, drone_b_speeds=drone_b_speeds,
                                                          num_trials=100)
plot_results(drone_b_speeds, results_drone_b_speed, 'Drone B Speed (m/s)', 'Success Rate (%)',
             'Success Rate vs. Drone B Speed')
