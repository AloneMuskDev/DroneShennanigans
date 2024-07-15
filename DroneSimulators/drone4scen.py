import numpy as np
import matplotlib.pyplot as plt


# Function to calculate the magnitude of a vector
def magnitude(vector):
    return np.sqrt(np.sum(vector ** 2, axis=1))


# Function to generate the initial positions and velocities for the drones
def setup_drones():
    drone_a_initial = np.array([50, 1, 50])
    drone_b_initial = np.array([45, 1, 45])
    target = np.array([100, 80, 400])

    direction_a = target - drone_a_initial
    direction_b = target - drone_b_initial

    return drone_a_initial, drone_b_initial, target, direction_a, direction_b


# Function to check if drones collide
def check_collision(drone_a_initial, velocity_a, drone_b_initial, velocity_b, max_time=15, steps=1000):
    t = np.linspace(0, max_time, steps)
    position_a = drone_a_initial + np.outer(t, velocity_a)
    position_b = drone_b_initial + np.outer(t, velocity_b)

    for i in range(steps):
        if np.linalg.norm(position_a[i] - position_b[i]) < 1:  # Assuming collision if distance < 1 unit
            return True, t[i]

    return False, None


# Function to run the scenario and plot results
def run_scenario(drone_a_speed, drone_b_speeds):
    drone_a_initial, drone_b_initial, target, direction_a, direction_b = setup_drones()

    magnitude_a = np.linalg.norm(direction_a)
    velocity_a = (direction_a / magnitude_a) * drone_a_speed

    success_rates = []

    for speed_b in drone_b_speeds:
        successful_intercepts = 0

        magnitude_b = np.linalg.norm(direction_b)
        velocity_b = (direction_b / magnitude_b) * speed_b

        collision_detected, _ = check_collision(drone_a_initial, velocity_a, drone_b_initial, velocity_b)
        if collision_detected:
            successful_intercepts += 1

        success_rates.append(successful_intercepts)

    plt.figure()
    plt.plot(drone_b_speeds, success_rates, marker='o')
    plt.xlabel('Drone B Speed (m/s)')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate of Drone B vs. Speed')
    plt.grid(True)
    plt.show()


# Define ranges and parameters for analysis
drone_a_speed = 30
drone_b_speeds = np.linspace(30, 70, 10)

# Run the scenario
run_scenario(drone_a_speed, drone_b_speeds)
