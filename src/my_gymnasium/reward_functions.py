import numpy as np
from config import DECEL_DISTANCE, PROXIMITY_THRESHOLD, TIME_PENALTY_THRESHOLD

def heading_reward(heading_diff):
    """
    Calculate heading reward using a modified Gaussian function:
    - Returns maximum reward (10.0) when heading_diff is 0
    - Drops off quickly after pi/4
    - Becomes negative after pi/2 (90 degrees)
    - Symmetric negative rewards for angles > 90 degrees
    """
    sigma = np.pi/6  # Standard deviation of ~30 degrees
    if heading_diff <= np.pi/2:
        # Normal Gaussian for angles <= 90 degrees
        return 10.0 * np.exp(-0.5 * (heading_diff/sigma)**2)
    else:
        # Mirror the Gaussian curve but make it negative for angles > 90 degrees
        mirrored_diff = np.pi - heading_diff  # Distance from 180 degrees
        return -10.0 * np.exp(-0.5 * (mirrored_diff/sigma)**2)


def obstacle_heading_reward(heading_diff):
    """
    Calculate heading reward for obstacle avoidance:
    - Returns negative reward when heading directly at obstacle (heading_diff near 0)
    - Returns positive reward for heading differences between pi/8 and pi/4
    - Returns gradually increasing positive reward for heading differences > pi/4
    """
    if heading_diff < np.pi/8:
        # Negative reward that peaks at heading_diff = 0
        return -3.0 * (1.0 - heading_diff/(np.pi/8))
    else:
        # Smooth positive reward curve that:
        # - Starts positive at pi/8
        # - Peaks at pi/2 (90 degrees)
        # - Gradually levels off after pi/2
        return 2.0 * (1.0 - np.exp(-(heading_diff - np.pi/8)))  # Exponential approach to maximum


def speed_reward(speed_ratio, distance_to_goal):
    """
    Calculate speed reward based on speed ratio and distance to goal:
    - For distances <= 10:
        - Negative rewards for speed_ratio > 1 or < 0.5
        - Positive rewards peaking at speed_ratio = 0.75 for 0.5 <= speed_ratio <= 1
    - For distances > 10:
        - Rewards speeds close to desired speed (speed_ratio ≈ 1)
        - Severely punishes very low speeds to prevent stalling
    """
    if distance_to_goal <= DECEL_DISTANCE:
        # For close distances, we want a bell curve centered at 0.75
        if speed_ratio < 0.5:
            # Quadratic penalty for low speeds
            return -10.0 * (0.5 - speed_ratio) ** 2
        elif speed_ratio > 1.0:
            # Linear penalty for high speeds
            return -5.0 * (speed_ratio - 1.0)
        else:
            # Bell curve centered at 0.75 for the sweet spot
            return 5.0 * np.exp(-((speed_ratio - 0.75) / 0.15) ** 2)
    else:
        # For larger distances, encourage high speeds
        if speed_ratio < 0.3:
            # Severe penalty for very low speeds (stalling)
            return -3.5 - 80.0 * (0.3 - speed_ratio) ** 2
        elif speed_ratio < 0.75:
            # Moderate penalty for suboptimal speeds
            return 3.5 - 5.0 * (1.0 - speed_ratio)
        else:
            # Reward for maintaining speed close to desired
            return 2.0 * np.exp(-((speed_ratio - 1.0) / 0.2) ** 2)


def proximity_reward(distance_to_goal):
    """
    Calculate proximity reward based on distance to goal:
    - Starts giving rewards when distance < PROXIMITY_THRESHOLD
    - Exponentially increases as distance decreases
    """
    if distance_to_goal >= PROXIMITY_THRESHOLD:
        return 0.0
    
    return 5.0 * np.exp(-2.0 * distance_to_goal / PROXIMITY_THRESHOLD)


def time_penalty(steps):
    """
    Calculate time-based penalty that grows progressively:
    - Starts small for early steps (-0.01)
    - Grows exponentially after threshold (500 steps)
    - Becomes severe near timeout (1000 steps)
    
    Parameters:
        steps (int): Current step count in the episode
    
    Returns:
        float: Negative reward that grows more negative with more steps
    """
    if steps <= TIME_PENALTY_THRESHOLD:
        return -0.01  # Small constant penalty for each step
    else:
        # Exponential growth of penalty after threshold
        normalized_steps = (steps - TIME_PENALTY_THRESHOLD) / (1000 - TIME_PENALTY_THRESHOLD)
        return  0.99 - np.exp(2.0 * normalized_steps)


def rotation_penalty(cumulative_rotation):
    """
    Calculate penalty for excessive rotation:
    - No penalty for rotations less than pi
    - Linear growth of penalty between pi and 2pi
    - Large penalty for full rotation (2pi)
    
    Parameters:
        cumulative_rotation (float): Total accumulated rotation in radians
    
    Returns:
        float: Negative reward that grows with rotation amount
    """
    if cumulative_rotation < np.pi:
        return 0.0
    elif cumulative_rotation < 2*np.pi:
        # Linear growth of penalty approaching full rotation
        normalized_rotation = (cumulative_rotation - np.pi) / np.pi
        return -10.0 * normalized_rotation
    else:
        # Severe penalty for full rotation
        return -50.0
