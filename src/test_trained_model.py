from stable_baselines3 import SAC
from environment import ASVEnvironment
from config import setup_logger, DEFAULT_SEED
import matplotlib.pyplot as plt
import time

# Setup logging
logger = setup_logger('ASV_Testing')

def test_trained_model(episodes=1, seed=43):
    """Test the trained model for multiple episodes"""
    
    # Create environment with same seed
    env = ASVEnvironment(seed=seed)
    
    # Load the trained model
    model = SAC.load("models/v7/sac_asv")
    
    for episode in range(episodes):
        logger.info(f"\nStarting Episode {episode + 1}")
        
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, truncated, _ = env.step(action)
            
            if done or truncated:
                if truncated:
                    logger.warning("Episode truncated - Out of bounds or collision!")
                elif done:
                    logger.info("Episode completed successfully!")
                break
                
            episode_reward += reward
            
            # Log state
            x, y = env.state[:2]
            logger.info(f"Step {step}: Action: {action}, Position: ({x:.2f}, {y:.2f}), Reward: {reward:.2f}")
            env.render()
            time.sleep(0.1)
            step += 1
            
        logger.info(f"Episode {episode + 1} finished after {step} steps. Total reward: {episode_reward:.2f}")
    
    # Keep the plot window open until manually closed
    plt.show()

if __name__ == "__main__":
    test_trained_model() 