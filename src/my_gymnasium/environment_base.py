import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces

from config import (
    ENV_HEIGHT,
    ENV_WIDTH,
    DECEL_DISTANCE,
    DECEL_FACTOR,
    GOAL_BOX_SIZE,
    MAX_ANGULAR_VELOCITY,
    MAX_TIMESTEPS,
    MAX_SPEED,
    TURN_RATE_FACTOR,
    setup_logger
)
from my_gymnasium.visualization import ASVVisualizer
from my_gymnasium.reward_functions import (
    heading_reward, 
    speed_reward, 
    proximity_reward,
    time_penalty,
)

logger = setup_logger('ASV_Base_Environment')

VERSION = "1.0.0"  # Increment this when making significant changes

class ASVBaseEnvironment(gym.Env):
    def __init__(self, seed=None):
        self.version = VERSION
        super().__init__()
        # Set the seed for the environment
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        self.action_space = spaces.Box(
            low=np.array([-1.0, -np.pi/4], dtype=np.float32),  # min(throttle), min(rudder)
            high=np.array([1.0, np.pi/4], dtype=np.float32),   # max(throttle), max(rudder)
            dtype=np.float32
        )
        
        # Simplified observation space without obstacles and threat zones
        self.observation_space = spaces.Dict({
            # Vessel state - normalized values
            'vessel_state': spaces.Box(
                low=np.array([-1.0, -1.0, -1.0, 0.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            ),
            # Goal information - normalized
            'goal_info': spaces.Box(
                low=np.array([0.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            ),
        })
        
        self.visualizer = ASVVisualizer()
        
        # Call reset to initialize the environment state
        self.reset(seed=seed)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        
        # Generate goal on the perimeter of a box (-GOAL_BOX_SIZE to GOAL_BOX_SIZE in x and y)
        # First decide which side of the box to place the goal on (0: top, 1: right, 2: bottom, 3: left)
        side = self.np_random.integers(0, 4)
        
        if side == 0:  # Top
            goal_x = self.np_random.uniform(-GOAL_BOX_SIZE, GOAL_BOX_SIZE)
            goal_y = GOAL_BOX_SIZE
        elif side == 1:  # Right
            goal_x = GOAL_BOX_SIZE
            goal_y = self.np_random.uniform(-GOAL_BOX_SIZE, GOAL_BOX_SIZE)
        elif side == 2:  # Bottom
            goal_x = self.np_random.uniform(-GOAL_BOX_SIZE, GOAL_BOX_SIZE)
            goal_y = -GOAL_BOX_SIZE
        else:  # Left
            goal_x = -GOAL_BOX_SIZE
            goal_y = self.np_random.uniform(-GOAL_BOX_SIZE, GOAL_BOX_SIZE)
            
        self.goal = np.array([goal_x, goal_y], dtype=np.float32)
        
        # Start closer to center
        start_x = self.np_random.uniform(-1.0, 1.0)
        start_y = self.np_random.uniform(-1.0, 1.0)
        
        # Calculate direction to goal
        to_goal = self.goal - np.array([start_x, start_y])
        goal_direction = np.arctan2(to_goal[1], to_goal[0])
        
        # Generate random heading within ±45 degrees of goal direction
        heading_offset = self.np_random.uniform(-np.pi/4, np.pi/4)  # ±45 degrees
        random_heading = goal_direction + heading_offset
        
        # Ensure heading is in [-π, π]
        random_heading = np.mod(random_heading + np.pi, 2*np.pi) - np.pi
        
        # Start with zero speed
        initial_speed = 0.0
        
        self.state = np.array([start_x, start_y, random_heading, initial_speed, 0.0], dtype=np.float32)
        
        # On reset, set last_distance_to_goal to current distance
        distance_to_goal = np.linalg.norm(self.goal - self.state[:2])
        self.last_distance_to_goal = distance_to_goal

        if self.visualizer is None:
            self.visualizer = ASVVisualizer()
        
        # Create initial observation
        observation = {
            'vessel_state': self.state,
            'goal_info': np.array([
                distance_to_goal,
                np.arctan2(self.goal[1], self.goal[0])
            ]),
        }
        
        return observation, {}
        
    def render(self):
        if self.visualizer is None:
            self.visualizer = ASVVisualizer()
            
        # Extract position and heading from state
        x, y, heading = self.state[:3]
        
        self.visualizer.draw_vessel(x, y, heading)
        self.visualizer.draw_goal(*self.goal)
        self.visualizer.render()
        
    def close(self):
        """Clean up resources and close visualization"""
        if self.visualizer is not None:
            plt.close(self.visualizer.fig)  # Close the matplotlib figure
            self.visualizer = None  # Remove the visualizer reference 
        
    def _normalize_observation(self, x, y, heading, speed, angular_velocity, distance_to_goal, goal_bearing):
        """Normalize observation values to [-1, 1] or [0, 1] range"""
        # Normalize positions to [-1, 1]
        norm_x = x / (ENV_WIDTH/2)
        norm_y = y / (ENV_HEIGHT/2)
        # Heading is already in [-π, π], normalize to [-1, 1]
        norm_heading = heading / np.pi
        # Speed to [0, 1]
        norm_speed = speed / MAX_SPEED
        # Angular velocity to [-1, 1]
        norm_angular_vel = angular_velocity / MAX_ANGULAR_VELOCITY
        # Distance to [0, 1]
        norm_distance = distance_to_goal / (ENV_WIDTH * np.sqrt(2))
        # Bearing is already in [-π, π], normalize to [-1, 1]
        norm_bearing = goal_bearing / np.pi
        
        return {
            'vessel_state': np.array([norm_x, norm_y, norm_heading, norm_speed, norm_angular_vel], dtype=np.float32),
            'goal_info': np.array([norm_distance, norm_bearing], dtype=np.float32)
        }

    def step(self, action):
        # Increment step counter
        self.steps += 1
        throttle, rudder = action
        x, y, heading, speed, angular_velocity = self.state
        dt = 0.2  # time step

        # Initialize done and truncated flags
        done = False
        truncated = False

        # Make turning less effective at higher speeds (more realistic vessel behavior)
        turn_effectiveness = 1.0 / (1.0 + (speed / MAX_SPEED))  # Decreases with speed
        angular_acceleration = TURN_RATE_FACTOR * rudder * turn_effectiveness
        
        # Update angular velocity and clip to maximum
        angular_velocity += angular_acceleration * dt
        angular_velocity = np.clip(angular_velocity, -MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)
        
        # Update heading
        heading += angular_velocity * dt
        heading = np.mod(heading + np.pi, 2*np.pi) - np.pi  # keep in [-π, π]
        
        # Update speed
        speed += throttle * dt
        speed = np.clip(speed, 0, MAX_SPEED)  # prevent negative speed
        
        # Update position
        x += speed * np.cos(heading) * dt
        y += speed * np.sin(heading) * dt
        
        # Update state
        self.state = np.array([x, y, heading, speed, angular_velocity])

        # Calculate distance to goal and goal bearing
        distance_to_goal = np.linalg.norm(self.goal - np.array([x, y]))
        goal_bearing = np.arctan2(self.goal[1] - y, self.goal[0] - x)

        self.last_distance_to_goal = distance_to_goal

        # Get normalized observations
        normalized_obs = self._normalize_observation(
            x, y, heading, speed, angular_velocity,
            distance_to_goal, goal_bearing
        )
        
        # Combine all observations
        observation = {**normalized_obs,}

        # Begin reward calculation
        reward = 0.0
        
        progress = self.last_distance_to_goal - distance_to_goal
        self.last_distance_to_goal = distance_to_goal
        if progress > 0:
            reward += progress
        else:
            reward -= progress

        # Calculate heading difference to goal
        heading_diff = abs(np.mod(goal_bearing - heading + np.pi, 2*np.pi) - np.pi)
        
        # Add heading reward
        reward += heading_reward(heading_diff)

        # 3. Progressive proximity reward
        reward += proximity_reward(distance_to_goal)

        # 4. Speed control for when near goal
        desired_speed = MAX_SPEED * (1.0 / (1.0 + np.exp(-DECEL_FACTOR * (distance_to_goal - DECEL_DISTANCE))))
        desired_speed = np.clip(desired_speed, 0.0, MAX_SPEED)
        speed_ratio = speed / desired_speed
        reward += speed_reward(speed_ratio, distance_to_goal)

        # Add progressive time penalty
        reward += time_penalty(self.steps)

        # Success check
        if distance_to_goal < 3.0:
            reward += 1000.0
            done = True
            logger.info(f"Successfully reached goal at position ({x:.1f}, {y:.1f})")
            return observation, reward, done, truncated, {}
        
        # Out of bounds check
        if abs(x) > ENV_WIDTH/2 or abs(y) > ENV_HEIGHT/2:
            truncated = True
            reward -= 50.0
            logger.warning(f"Out of bounds at position ({x:.2f}, {y:.2f})")
            return observation, reward, done, truncated, {}
        
        if self.steps > MAX_TIMESTEPS:
            truncated = True
            reward -= 100.0
            logger.warning("Episode truncated due to timeout")
            return observation, reward, done, truncated, {}

        return observation, reward, done, truncated, {}

