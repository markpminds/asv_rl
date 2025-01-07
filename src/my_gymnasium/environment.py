import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces

from config import (
    ENV_HEIGHT,
    ENV_WIDTH,
    N_OBSTACLES,
    N_THREAT_ZONES,
    MAX_ANGULAR_VELOCITY,
    MAX_SPEED,
    TURN_RATE_FACTOR,
    setup_logger
)
from my_gymnasium.visualization import ASVVisualizer

logger = setup_logger('ASV_Environment')

VERSION = "1.0.0"  # Increment this when making significant changes

class ASVEnvironment(gym.Env):
    def __init__(self, seed=None):
        self.version = VERSION
        super().__init__()
        # Set the seed for the environment
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        self.action_space = spaces.Box(
            low=np.array([0.0, -np.pi/8], dtype=np.float32),  # throttle, rudder
            high=np.array([1.0, np.pi/8], dtype=np.float32),  # throttle, rudder
            dtype=np.float32
        )
        
        # Normalize observation spaces to reasonable ranges
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
            'obstacles': spaces.Box(
                low=np.array([[0.0, 0.0] for _ in range(N_OBSTACLES)], dtype=np.float32),
                high=np.array([[1.0, 1.0] for _ in range(N_OBSTACLES)], dtype=np.float32),
                shape=(N_OBSTACLES, 2),
                dtype=np.float32
            ),
            'threat_zones': spaces.Box(
                low=np.array([[0.0, 0.0, 0.0, 0.0] for _ in range(N_THREAT_ZONES)], dtype=np.float32),
                high=np.array([[1.0, 1.0, 1.0, 1.0] for _ in range(N_THREAT_ZONES)], dtype=np.float32),
                shape=(N_THREAT_ZONES, 4),
                dtype=np.float32
            )
        })
        
        self.visualizer = ASVVisualizer()
        
        # Call reset to initialize the environment state
        self.reset(seed=seed)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Generate random goal first (since obstacles and zones need to avoid it)
        while True:
            goal_x = self.np_random.uniform(-ENV_WIDTH/2, ENV_WIDTH/2)
            goal_y = self.np_random.uniform(-ENV_HEIGHT/2, ENV_HEIGHT/2)
            self.goal = np.array([goal_x, goal_y], dtype=np.float32)
            
            # Ensure goal is not too close to starting area
            if np.linalg.norm(self.goal) > 40.0:  # Keep goal away from center
                break
        
        # Generate random obstacles
        self.obstacles = []
        for _ in range(N_OBSTACLES):  # Use constant from config
            while True:
                x = self.np_random.uniform(-ENV_WIDTH/2 * 0.8, ENV_WIDTH/2 * 0.8)
                y = self.np_random.uniform(-ENV_HEIGHT/2 * 0.8, ENV_HEIGHT/2 * 0.8)
                radius = self.np_random.uniform(3.0, 7.0)
                
                # Check if obstacle overlaps with goal or starting area
                obstacle_pos = np.array([x, y])
                if (np.linalg.norm(obstacle_pos - self.goal) > radius + 10.0 and  # Clear of goal
                    np.linalg.norm(obstacle_pos) > radius + 10.0):                 # Clear of start
                    self.obstacles.append([x, y, radius])
                    break
        
        # Generate random threat zones
        self.threat_zones = []
        for _ in range(N_THREAT_ZONES):  # Use constant from config
            while True:
                x = self.np_random.uniform(-ENV_WIDTH/2 * 0.8, ENV_WIDTH/2 * 0.8)
                y = self.np_random.uniform(-ENV_HEIGHT/2 * 0.8, ENV_HEIGHT/2 * 0.8)
                width = self.np_random.uniform(10.0, 20.0)
                height = self.np_random.uniform(10.0, 20.0)
                
                # Check if threat zone overlaps with goal or starting area
                if (abs(x - self.goal[0]) > width/2 + 10.0 and abs(y - self.goal[1]) > height/2 + 10.0 and
                    abs(x) > width/2 + 10.0 and abs(y) > height/2 + 10.0):
                    self.threat_zones.append([x, y, width, height])
                    break
        
        # Start closer to center
        start_x = self.np_random.uniform(-5.0, 5.0)
        start_y = self.np_random.uniform(-5.0, 5.0)
        
        # Start with heading roughly toward goal
        goal_direction = np.arctan2(self.goal[1] - start_y, self.goal[0] - start_x)
        random_heading = goal_direction + self.np_random.uniform(-np.pi/4, np.pi/4)  # ±45 degrees
        
        # Start with small positive speed
        initial_speed = 2.0  # Always start moving
        
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
            'obstacles': self._get_obstacle_observations(),
            'threat_zones': self._get_threat_zone_observations()
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
        # Add small random noise to actions
        action_noise = self.np_random.normal(0, 0.05, size=2)  # 5% noise
        noisy_action = np.clip(action + action_noise, 
                              self.action_space.low, 
                              self.action_space.high)
        
        throttle, rudder = noisy_action
        x, y, heading, speed, angular_velocity = self.state
        
        dt = 1  # time step

        # Initialize done and truncated flags
        done = False
        truncated = False

        # Scaled model: angular_acceleration only depends on rudder angle
        angular_acceleration = TURN_RATE_FACTOR * rudder  # Removed speed dependency
        
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

        reward = 0.0
        distance_to_goal = np.linalg.norm(self.goal - np.array([x, y]))
        goal_bearing = np.arctan2(self.goal[1] - y, self.goal[0] - x)

        if self.last_distance_to_goal is not None:
            # 1. Strong progress reward
            progress = self.last_distance_to_goal - distance_to_goal
            reward += progress * 3.0  # Base progress reward
            
            # 2. heading reward
            heading_diff = abs(np.mod(goal_bearing - heading + np.pi, 2*np.pi) - np.pi)
            
            # Stronger rewards for good heading
            if heading_diff < np.pi/6:  # Within 30 degrees
                reward += 2.0
            elif heading_diff > np.pi/2:  # More than 90 degrees off
                reward -= 2.0
            
        self.last_distance_to_goal = distance_to_goal

        # Get normalized observations first
        normalized_obs = self._normalize_observation(
            x, y, heading, speed, angular_velocity,
            distance_to_goal, goal_bearing
        )
        
        # Get obstacle and threat zone observations
        # obstacle_obs = self._get_obstacle_observations()
        # threat_zone_obs = self._get_threat_zone_observations()
        
        # Combine all observations
        observation = {
            **normalized_obs,
            #'obstacles': obstacle_obs,
            #'threat_zones': threat_zone_obs
        }

        # Check for collisions and threat zone violations
        # Obstacle collision check
        """
        for i, (distance, radius) in enumerate(obstacle_obs):
            if distance < radius:  # Collision occurred
                reward -= 25.0
                truncated = True
                logger.warning(f"Collision with obstacle {i+1} at position ({self.obstacles[i][0]:.1f}, {self.obstacles[i][1]:.1f})")
                return observation, reward, done, truncated, {}

        # Threat zone check
        for i, (distance, _, _, is_inside) in enumerate(threat_zone_obs):
            if is_inside > 0.5:  # Inside threat zone (is_inside is a float)
                reward -= 50.0
                truncated = True
                logger.warning(f"Entered threat zone {i+1} at position ({x:.1f}, {y:.1f})")
                return observation, reward, done, truncated, {}
        
        """

        # Success check
        if distance_to_goal < 5.0:
            reward += 100.0
            done = True
            logger.info(f"Successfully reached goal at position ({x:.1f}, {y:.1f})")
        
        # Out of bounds check
        if abs(x) > ENV_WIDTH/2 or abs(y) > ENV_HEIGHT/2:
            reward -= 25.0
            truncated = True
            logger.warning(f"Out of bounds at position ({x:.2f}, {y:.2f})")

        info = {
            'distance_to_goal': distance_to_goal,
            'heading_error': heading_diff,
            'speed': speed,
            'is_success': done  # True if reached goal
        }
        
        return observation, reward, done, truncated, info
        

    def _get_obstacle_observations(self):
        """Convert obstacle positions to distance observations"""
        obstacle_obs = np.zeros((N_OBSTACLES, 2), dtype=np.float32)
        
        x, y, _ = self.state[:3]
        vessel_pos = np.array([x, y])
        
        for i, obstacle in enumerate(self.obstacles):
            obs_x, obs_y, radius = obstacle
            obs_pos = np.array([obs_x, obs_y])
            
            # Calculate distance
            distance = np.linalg.norm(obs_pos - vessel_pos)
            
            obstacle_obs[i] = [distance, radius]
        
        return obstacle_obs
    
    def _get_threat_zone_observations(self):
        """Convert threat zone positions to distance observations"""
        threat_zone_obs = np.zeros((N_THREAT_ZONES, 4), dtype=np.float32)
        
        x, y, _ = self.state[:3]
        vessel_pos = np.array([x, y])

        for i, zone in enumerate(self.threat_zones):
            zone_x, zone_y, width, height = zone
            zone_center = np.array([zone_x, zone_y])
            
            # Calculate distance
            distance = np.linalg.norm(zone_center - vessel_pos)
            
            # Check if vessel is inside threat zone
            is_inside = (abs(x - zone_x) < width/2 and abs(y - zone_y) < height/2)
            
            threat_zone_obs[i] = [distance, width, height, float(is_inside)]
        
        return threat_zone_obs
