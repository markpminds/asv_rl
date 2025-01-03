import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces

from config import (  # Import all constants
    ENV_HEIGHT,
    ENV_WIDTH,
    GOAL,
    MAX_ANGULAR_VELOCITY,
    MAX_OBS_RADIUS,
    MAX_SPEED,
    N_THREAT_ZONES,
    OBSTACLES,
    TURN_RATE_FACTOR,
    THREAT_ZONES,
)
from visualization import ASVVisualizer


class ASVEnvironment(gym.Env):
    def __init__(self):
        super().__init__()
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-0.25, -np.pi/4]),  # [throttle (directly maps to acceleration in m/s^2), rudder_angle]
            high=np.array([1.0, np.pi/4]),    # Limit rudder to ±45 degrees
            dtype=np.float64
        )
        
        self.observation_space = spaces.Dict({
            # Vessel state
            'vessel_state': spaces.Box(
                low=np.array([-ENV_WIDTH/2, -ENV_HEIGHT/2, -np.pi, 0.0, -np.pi/2]),
                high=np.array([ENV_WIDTH/2, ENV_HEIGHT/2, np.pi, 18.0, np.pi/2]),
                dtype=np.float64
            ),
            # Goal information
            'goal_info': spaces.Box(
                low=np.array([0.0, -np.pi]),    # [distance_to_goal, relative_bearing]
                high=np.array([ENV_WIDTH * (2 ** 0.5), np.pi]),
                dtype=np.float64
            ),
            # Obstacle information
            'obstacles': spaces.Box(
                low=np.array([[0.0, -np.pi, 0.0] for _ in range(len(OBSTACLES))]),    # [distance, bearing, radius]
                high=np.array([[ENV_WIDTH * (2 ** 0.5), np.pi, 100.0] for _ in range(len(OBSTACLES))]),
                shape=(len(OBSTACLES), 3),
                dtype=np.float64
            ),
            # Threat zone information
            'threat_zones': spaces.Box(
                low=np.array([[0.0, -np.pi, 0.0, 0.0] for _ in range(len(THREAT_ZONES))]),    
                high=np.array([[ENV_WIDTH * (2 ** 0.5), np.pi, ENV_WIDTH, ENV_HEIGHT] 
                              for _ in range(len(THREAT_ZONES))]),
                shape=(len(THREAT_ZONES), 4),  # [distance_to_center, bearing_to_center, width, height]
                dtype=np.float64
            )
        })
        
        # Initialize visualization
        self.visualizer = None
        self.state = None
        self.obstacles = OBSTACLES
        self.goal = GOAL  # Goal position [x, y]
        self.threat_zones = THREAT_ZONES
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize state: [x, y, heading, speed, angular_velocity]
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        if self.visualizer is None:
            self.visualizer = ASVVisualizer()
        
        # Create initial observation
        observation = {
            'vessel_state': self.state,
            'goal_info': np.array([
                np.linalg.norm(self.goal - self.state[:2]),  # distance to goal
                np.arctan2(self.goal[1], self.goal[0])  # initial bearing to goal
            ]),
            'obstacles': self._get_obstacle_observations()
        }
        
        return observation, {}
        
    def render(self):
        if self.visualizer is None:
            self.visualizer = ASVVisualizer()
            
        # Extract position and heading from state
        x, y, heading = self.state[:3]
        
        # Draw everything
        self.visualizer.draw_vessel(x, y, heading)
        self.visualizer.draw_obstacles(self.obstacles)
        self.visualizer.draw_threat_zones(self.threat_zones)
        self.visualizer.draw_goal(*self.goal)
        self.visualizer.render() 
        
    def close(self):
        """Clean up resources and close visualization"""
        if self.visualizer is not None:
            plt.close(self.visualizer.fig)  # Close the matplotlib figure
            self.visualizer = None  # Remove the visualizer reference 
        
    def step(self, action):
        # Unpack state and action
        throttle, rudder = action
        x, y, heading, speed, angular_velocity = self.state
        
        dt = 1  # time step
        
        # ------- Constants -------

        # Scaled model: angular_acceleration proportional to rudder angle and speed
        angular_acceleration = TURN_RATE_FACTOR * rudder * speed  # rad/s²
        
        # Update angular velocity and clip to maximum
        angular_velocity += angular_acceleration * dt
        angular_velocity = np.clip(angular_velocity, -MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)
        
        # Update heading
        heading += angular_velocity * dt
        heading = np.mod(heading + np.pi, 2*np.pi) - np.pi  # keep in [-π, π]
        
        # Simple acceleration from throttle
        speed += throttle * dt
        speed = np.clip(speed, 0, MAX_SPEED)
        
        # Update position
        x += speed * np.cos(heading) * dt
        y += speed * np.sin(heading) * dt
        
        # Update state
        self.state = np.array([x, y, heading, speed, angular_velocity])
        
        # -------------- Reward and termination logic (example) --------------
        distance_to_goal = np.linalg.norm(self.goal - np.array([x, y]))
        reward = -distance_to_goal
        
        # Mild penalty for high angular velocity
        reward -= 0.1 * abs(angular_velocity)
        
        done = False
        if distance_to_goal < 1.0:
            done = True
            reward += 100  # bonus for reaching goal
        
        if abs(x) > 1000 or abs(y) > 1000:
            done = True
            reward -= 100  # out-of-bounds penalty
        
        # Create or update your observation dictionary here
        observation = {
            'vessel_state': np.array([x, y, heading, speed, angular_velocity]),
            'goal_info': np.array([
                distance_to_goal,
                np.arctan2(self.goal[1] - y, self.goal[0] - x) - heading  # relative bearing
            ]),
            'obstacles': self._get_obstacle_observations()
        }
        
        return observation, reward, done, False, {}
        
    def _get_obstacle_observations(self):
        """Convert obstacle positions to distance/bearing observations"""
        N_OBSTACLES = len(self.obstacles)
        obstacle_obs = np.zeros((N_OBSTACLES, 3))
        
        x, y, heading = self.state[:3]
        vessel_pos = np.array([x, y])
        
        for i, obstacle in enumerate(self.obstacles):
            obs_x, obs_y, radius = obstacle
            obs_pos = np.array([obs_x, obs_y])
            
            # Calculate distance
            distance = np.linalg.norm(obs_pos - vessel_pos)
            
            # Calculate relative bearing
            bearing = np.arctan2(obs_y - y, obs_x - x) - heading
            bearing = np.mod(bearing + np.pi, 2 * np.pi) - np.pi  # Normalize to [-π, π]
            
            obstacle_obs[i] = [distance, bearing, radius]
        
        return obstacle_obs 