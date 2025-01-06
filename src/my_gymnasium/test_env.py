from environment import ASVEnvironment
from config import setup_logger
import time
import matplotlib.pyplot as plt

logger = setup_logger('ASV_Environment_Test')

def test_environment():
    # Create and initialize environment
    env = ASVEnvironment()
    
    # Get initial state
    obs, _ = env.reset()
    
    # Get actual position from environment state (not normalized)
    actual_x, actual_y = env.state[:2]
    
    logger.info("Initial State:")
    logger.info(f"Vessel Position: ({actual_x:.2f}, {actual_y:.2f})")  # Use actual state values
    logger.info(f"Vessel Heading: {env.state[2]:.2f}")  # Actual heading in radians
    logger.info(f"Goal Position: ({env.goal[0]:.2f}, {env.goal[1]:.2f})")
    logger.info("Obstacles:")
    for i, obs in enumerate(env.obstacles):
        logger.info(f"Obstacle {i+1}: position ({obs[0]:.2f}, {obs[1]:.2f}), radius {obs[2]:.2f}")
    logger.info("Threat Zones:")
    for i, zone in enumerate(env.threat_zones):
        logger.info(f"Zone {i+1}: center ({zone[0]:.2f}, {zone[1]:.2f}), width {zone[2]:.2f}, height {zone[3]:.2f}")

    # Visualize environment
    logger.info("Displaying environment visualization...")
    env.render()
    
    # Test a few random actions
    logger.info("Testing random actions:")
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, done, truncated, _ = env.step(action)
        
        # Get actual position from environment state
        actual_x, actual_y = env.state[:2]
        
        logger.info(f"Step {i+1}:")
        logger.info(f"Action: throttle={action[0]:.2f}, rudder={action[1]:.2f}")
        logger.info(f"Position: ({actual_x:.2f}, {actual_y:.2f})")  # Use actual state values
        logger.info(f"Reward: {reward:.2f}")
        
        env.render()
        time.sleep(0.1)  # Pause to make visualization visible
        
        if done or truncated:
            logger.info("Episode ended!")
            break
    
    # Keep the plot window open until manually closed
    plt.show()

if __name__ == "__main__":
    test_environment() 