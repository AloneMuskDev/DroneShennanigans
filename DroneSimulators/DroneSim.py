import numpy as np
import random


# Function to generate random position vector within given ranges
def random_position():
    x = round(random.uniform(0, 500))
    y = round(random.uniform(0, 120))
    z = round(random.uniform(0, 500))
    return np.array([x, y, z])


# Function to generate random velocity vector with a reasonable speed
def random_velocity():
    speed = random.uniform(1, 10)  # Reasonable speed range
    direction = np.random.rand(3) - 0.5  # Random direction
    direction /= np.linalg.norm(direction)  # Normalize direction
    velocity = direction * speed
    return np.round(velocity)


# Function to calculate the time at which Drone A reaches the target
def time_to_target(position, velocity, target):
    distance = np.linalg.norm(target - position)
    speed = np.linalg.norm(velocity)
    return distance / speed


# Function to find the time at which the drones are at a minimum distance
def time_of_minimum_distance(p1, v1, p2, v2):
    # Relative position and velocity
    r = p1 - p2
    v = v1 - v2
    v_dot_v = np.dot(v, v)

    if v_dot_v == 0:
        return 0  # If the velocities are the same, set t_min to 0

    t_min = -np.dot(r, v) / v_dot_v
    return max(t_min, 0)


# Function to check if a collision occurs
def check_collision(drone_a_position, drone_a_velocity, drone_b_position, drone_b_velocity):
    t_min_distance = time_of_minimum_distance(drone_a_position, drone_a_velocity, drone_b_position, drone_b_velocity)

    # Positions at time of minimum distance
    pos_a_at_t_min = drone_a_position + drone_a_velocity * t_min_distance
    pos_b_at_t_min = drone_b_position + drone_b_velocity * t_min_distance

    min_distance = np.linalg.norm(pos_a_at_t_min - pos_b_at_t_min)

    return t_min_distance, min_distance


# Function to find a scenario where the collision is successful
def find_successful_collision():
    while True:
        drone_a_position = random_position()
        drone_b_position = random_position()

        # Ensure Drone B starts from a different position
        while np.array_equal(drone_a_position, drone_b_position):
            drone_b_position = random_position()

        # Velocity of Drone A directed towards the target
        direction_to_target = target - drone_a_position
        direction_to_target = direction_to_target.astype(float)  # Convert to float array for division
        direction_to_target /= np.linalg.norm(direction_to_target)
        speed_a = random.uniform(5, 15)  # Reasonable speed range for Drone A
        drone_a_velocity = np.round(direction_to_target * speed_a)

        # Random velocity for Drone B
        speed_b = random.uniform(5, 15)  # Reasonable speed range for Drone B
        drone_b_velocity = random_velocity()

        # Check collision
        t_min_distance, min_distance = check_collision(drone_a_position, drone_a_velocity, drone_b_position,
                                                       drone_b_velocity)

        # If collision occurs
        if min_distance == 0:
            return drone_a_position, drone_a_velocity, drone_b_position, drone_b_velocity, t_min_distance


# Target position
target = np.array([100, 400, 80])

# Number of scenarios to investigate
num_scenarios = 10

# Investigate scenarios
for _ in range(num_scenarios):
    # Random starting positions
    drone_a_position = random_position()
    drone_b_position = random_position()

    # Ensure Drone B starts from a different position
    while np.array_equal(drone_a_position, drone_b_position):
        drone_b_position = random_position()

    # Velocity of Drone A directed towards the target
    direction_to_target = target - drone_a_position
    direction_to_target = direction_to_target.astype(float)  # Convert to float array for division
    direction_to_target /= np.linalg.norm(direction_to_target)
    speed_a = random.uniform(5, 15)  # Reasonable speed range for Drone A
    drone_a_velocity = np.round(direction_to_target * speed_a)

    # Random velocity for Drone B
    speed_b = random.uniform(15, 30)  # Reasonable speed range for Drone B
    drone_b_velocity = random_velocity()

    # Time for Drone A to reach the target
    t_a_to_target = time_to_target(drone_a_position, drone_a_velocity, target)

    # Check collision
    t_min_distance, min_distance = check_collision(drone_a_position, drone_a_velocity, drone_b_position,
                                                   drone_b_velocity)

    # Determine if Drone B is successful
    successful_intercept = t_min_distance < t_a_to_target and min_distance == 0  # Collision if distance is zero

    print(f"Scenario {_ + 1}")
    print(f"Drone A initial position: {drone_a_position}, velocity: {drone_a_velocity}")
    print(f"Drone B initial position: {drone_b_position}, velocity: {drone_b_velocity}")
    print(f"Time for Drone A to reach the target: {t_a_to_target:.2f} seconds")
    print(f"Time of minimum distance: {t_min_distance:.2f} seconds")
    print(f"Minimum distance: {min_distance:.2f} units")

    if successful_intercept:
        print("Drone B successfully intercepted Drone A.")
    else:
        print("Drone B failed to intercept Drone A.")

    print("-" * 50)

# Find and print a successful collision scenario
drone_a_position, drone_a_velocity, drone_b_position, drone_b_velocity, t_min_distance = find_successful_collision()
print("Successful Collision Scenario:")
print(f"Drone A initial position: {drone_a_position}, velocity: {drone_a_velocity}")
print(f"Drone B initial position: {drone_b_position}, velocity: {drone_b_velocity}")
print(f"Time of minimum distance: {t_min_distance:.2f} seconds")
print(f"Minimum distance: 0 units (Collision)")
print("-" * 50)
