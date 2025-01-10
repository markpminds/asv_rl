import numpy as np
import logging

ENV_WIDTH = 100
ENV_HEIGHT = 100 

N_OBSTACLES = 1
N_THREAT_ZONES = 3

GOAL_BOX_SIZE = 40

DECEL_DISTANCE = 10 # Distance at which to start significant deceleration
DECEL_FACTOR = 0.2 # Controls how sharply the speed drops

OBSTACLE_AVOIDANCE_ZONE = 3

MAX_SPEED = 5.0
MAX_ANGULAR_VELOCITY = np.pi/8  # Reduced from π/4 for less aggressive turns
TURN_RATE_FACTOR = 0.1

PROXIMITY_THRESHOLD = 30.0
TIME_PENALTY_THRESHOLD = 500
MAX_TIMESTEPS = 1000

DEFAULT_SEED = 42
N_ENVS = 8
# Logging configuration
def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler('logs/simulation.log')
    
    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(log_format)
    f_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger 