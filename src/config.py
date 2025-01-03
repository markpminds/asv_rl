import numpy as np

# Environment dimensions
ENV_WIDTH = 2000
ENV_HEIGHT = 2000

GOAL = [1000, 1000]

# Obstacles configuration
OBSTACLES = [
    [500, 500, 100],
    [500, -500, 100],
    [-500, 500, 100],
    [-500, -500, 100]
]
MAX_OBS_RADIUS = 100

# Vessel dynamics
MAX_SPEED = 18.0  # meters/second
MAX_ANGULAR_VELOCITY = np.pi/2
TURN_RATE_FACTOR = 0.2

# Other constants
N_THREAT_ZONES = 3  # should be squares instead of circles 

# Threat Zones configuration
THREAT_ZONES = [
    # [center_x, center_y, width, height]
    [-200, 200, 100, 200],   # Example threat zone
    [300, -400, 250, 150],
    [-600, -300, 200, 200],
    [200, 600, 100, 300]
] 