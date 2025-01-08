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
            )
            # 'obstacles': spaces.Box(
            #     low=np.array([[0.0, 0.0] for _ in range(N_OBSTACLES)], dtype=np.float32),
            #     high=np.array([[1.0, 1.0] for _ in range(N_OBSTACLES)], dtype=np.float32),
            #     shape=(N_OBSTACLES, 2),
            #     dtype=np.float32
            # ),
            # 'threat_zones': spaces.Box(
            #     low=np.array([[0.0, 0.0, 0.0, 0.0] for _ in range(N_THREAT_ZONES)], dtype=np.float32),
            #     high=np.array([[1.0, 1.0, 1.0, 1.0] for _ in range(N_THREAT_ZONES)], dtype=np.float32),
            #     shape=(N_THREAT_ZONES, 4),
            #     dtype=np.float32
            # )
        })
        
        self.visualizer = ASVVisualizer()
        
        # Call reset to initialize the environment state
        self.reset(seed=seed)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
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
        
        # Generate one obstacle between start and goal
        self.obstacles = []
        # Calculate a point between start (around origin) and goal
        obstacle_distance = self.np_random.uniform(0.3, 0.7)  # How far along the path to place obstacle
        # Since start is near origin, we can use goal position directly
        obstacle_x = goal_x * obstacle_distance
        obstacle_y = goal_y * obstacle_distance
        # Add some random perpendicular offset
        perpendicular_x = -goal_y / np.linalg.norm(self.goal)  # Normalized perpendicular vector
        perpendicular_y = goal_x / np.linalg.norm(self.goal)
        offset = self.np_random.uniform(-5.0, 5.0)
        obstacle_x += perpendicular_x * offset
        obstacle_y += perpendicular_y * offset
        # Add the obstacle with random radius
        radius = self.np_random.uniform(3.0, 5.0)
        self.obstacles.append([obstacle_x, obstacle_y, radius])
        
        # # Generate random threat zones
        # self.threat_zones = []
        # for _ in range(N_THREAT_ZONES):  # Use constant from config
        #     while True:
        #         x = self.np_random.uniform(-ENV_WIDTH/2 * 0.8, ENV_WIDTH/2 * 0.8)
        #         y = self.np_random.uniform(-ENV_HEIGHT/2 * 0.8, ENV_HEIGHT/2 * 0.8)
        #         width = self.np_random.uniform(10.0, 20.0)
        #         height = self.np_random.uniform(10.0, 20.0)
        #         
        #         # Check if threat zone overlaps with goal or starting area
        #         if (abs(x - self.goal[0]) > width/2 + 10.0 and abs(y - self.goal[1]) > height/2 + 10.0 and
        #             abs(x) > width/2 + 10.0 and abs(y) > height/2 + 10.0):
        #             self.threat_zones.append([x, y, width, height])
        #             break
        
        # Start closer to center
        start_x = self.np_random.uniform(-1.0, 1.0)
        start_y = self.np_random.uniform(-1.0, 1.0)
        
        # Completely random initial heading
        random_heading = self.np_random.uniform(-np.pi, np.pi)
        
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
            # 'obstacles': self._get_obstacle_observations(),
            # 'threat_zones': self._get_threat_zone_observations()
        }
        
        return observation, {}
        
    def render(self):
        if self.visualizer is None:
            self.visualizer = ASVVisualizer()
            
        # Extract position and heading from state
        x, y, heading = self.state[:3]
        
        # Draw everything
        self.visualizer.draw_vessel(x, y, heading)
        # self.visualizer.draw_obstacles(self.obstacles)
        # self.visualizer.draw_threat_zones(self.threat_zones)
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

    def _heading_reward(self, heading_diff):
        """
        Calculate heading reward using a Gaussian-like function:
        - Returns maximum reward (2.0) when heading_diff is 0
        - Drops off quickly after pi/4
        - Returns negative values for heading differences larger than pi/2
        """
        sigma = np.pi/6  # Standard deviation of ~30 degrees
        reward = 2.0 * np.exp(-0.5 * (heading_diff/sigma)**2)  # Gaussian peak at 2.0
        
        # Add negative reward for very bad alignments (more than 90 degrees off)
        if heading_diff > np.pi/2:
            reward -= 2.0
            
        return reward

    def step(self, action):
        # Add small random noise to actions
        action_noise = self.np_random.normal(0, 0.05, size=2)  # 5% noise
        noisy_action = np.clip(action + action_noise, 
                              self.action_space.low, 
                              self.action_space.high)
        
        throttle, rudder = noisy_action
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
        
        # Allow for deceleration with negative throttle
        speed += throttle * dt
        speed = np.clip(speed, 0, MAX_SPEED)  # prevent negative speed
        
        # Update position
        x += speed * np.cos(heading) * dt
        y += speed * np.sin(heading) * dt
        
        # Update state
        self.state = np.array([x, y, heading, speed, angular_velocity])

        reward = 0.0
        distance_to_goal = np.linalg.norm(self.goal - np.array([x, y]))
        goal_bearing = np.arctan2(self.goal[1] - y, self.goal[0] - x)

        # 1. Progress reward
        progress = self.last_distance_to_goal - distance_to_goal
        reward += progress
        
        # 2. Heading reward using smooth function
        heading_diff = abs(np.mod(goal_bearing - heading + np.pi, 2*np.pi) - np.pi)
        heading_reward = self._heading_reward(heading_diff)
        reward += heading_reward

        # Speed control based on alignment and distance
        # if heading_reward > 1.0:  # Well aligned
        #    if distance_to_goal > 20.0 and speed > MAX_SPEED * 0.7:
        #        reward += 1.0  # Reward high speed when far and aligned
        #    elif distance_to_goal < 20.0 and speed < MAX_SPEED * 0.5:
        #        reward += 1.0  # Reward lower speed when close and aligned
        # elif heading_reward < -1.0:  # Badly aligned
        #     if speed > MAX_SPEED * 0.5:
        #        reward -= 1.0  # Penalty for high speed when badly aligned
        
        # 3. Speed control rewards
        # if distance_to_goal < 20.0 and speed > MAX_SPEED * 0.7:
        #    reward -= 1.0  # Penalty for excessive speed near goal
        
        # 3. Speed control reward based on distance to goal
        # Calculate desired speed using a sigmoid-like function that drops off more sharply
        # This creates a curve where speed stays high until DECEL_DISTANCE units away, then drops quickly
        desired_speed = MAX_SPEED * (1.0 / (1.0 + np.exp(-DECEL_FACTOR * (distance_to_goal - DECEL_DISTANCE))))
        desired_speed = np.clip(desired_speed, 0.0, MAX_SPEED)

        speed_ratio = speed / desired_speed
        
        # Calculate speed difference and apply penalties using walrus operator
        if distance_to_goal < DECEL_DISTANCE:
            if speed_ratio > 1:
                reward -= 3.0 * speed_ratio
            elif speed_ratio < 0.5:
                reward -= 1.0 * (1 - speed_ratio)
            else:
                reward += 2.0 * (1 - speed_ratio)

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
        if distance_to_goal < 3.0:
            reward += 100.0
            done = True
            logger.info(f"Successfully reached goal at position ({x:.1f}, {y:.1f})")
        
        # Out of bounds check
        if abs(x) > ENV_WIDTH/2 or abs(y) > ENV_HEIGHT/2:
            reward -= 25.0
            truncated = True
            logger.warning(f"Out of bounds at position ({x:.2f}, {y:.2f})")
        
        return observation, reward, done, truncated, {}
        

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
