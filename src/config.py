import numpy as np
import logging

ENV_WIDTH = 100
ENV_HEIGHT = 100 

N_OBSTACLES = 4
N_THREAT_ZONES = 3

MAX_SPEED = 10.0
MAX_ANGULAR_VELOCITY = np.pi/4  
TURN_RATE_FACTOR = 0.05 

DEFAULT_SEED = 42

# Original configuration (commented out)
"""
# Environment dimensions
# ENV_WIDTH = 200
# ENV_HEIGHT = 200

# GOAL = [75, 50]

# Obstacles configuration
# OBSTACLES = [
#     # [x, y, radius]
#     [50, 50, 10],
#     [50, -50, 10],
#     [-50, 50, 10],
#     [-50, -50, 10]
# ]

# Vessel dynamics
# MAX_SPEED = 18.0  # meters/second
# MAX_ANGULAR_VELOCITY = np.pi/2  # radians/second
# TURN_RATE_FACTOR = 0.1  # reduced from 0.2 to make turning slower

# Threat Zones configuration
# THREAT_ZONES = [
#     # [center_x, center_y, width, height]
#     [-20, 20, 10, 20],
#     [30, -40, 25, 15],
#     [-60, -30, 20, 20],
#     [20, 60, 10, 30]
# ]
"""

# Logging configuration
def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler('simulation.log')
    
    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(log_format)
    f_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger 