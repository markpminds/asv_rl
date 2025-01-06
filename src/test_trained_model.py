from stable_baselines3 import SAC, PPO
import argparse
import os
from my_gymnasium.environment import ASVEnvironment
from config import setup_logger, DEFAULT_SEED
import matplotlib.pyplot as plt
import time

# Setup logging
logger = setup_logger('ASV_Testing')

def get_model_class(model_type):
    """Get the model class based on model type string"""
    model_types = {
        'sac': SAC,
        'ppo': PPO
    }
    if model_type.lower() not in model_types:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: {list(model_types.keys())}")
    return model_types[model_type.lower()]

def test_trained_model(model_type='sac', version='v1', episodes=1, seed=DEFAULT_SEED):
    """Test the trained model for multiple episodes"""
    
    # Create environment with same seed
    env = ASVEnvironment(seed=seed)
    
    # Construct model path
    model_path = os.path.join("models", model_type, version, f"{model_type}_asv_final")
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model not found at {model_path}.zip")
    
    # Load the appropriate model type
    ModelClass = get_model_class(model_type)
    model = ModelClass.load(model_path)
    
    logger.info(f"Testing {model_type.upper()} model version {version}")
    
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
    parser = argparse.ArgumentParser(description='Test a trained RL model')
    parser.add_argument('--model-type', type=str, default='sac',
                      help='Type of model to test (sac, ppo)')
    parser.add_argument('--version', type=str, default='v1',
                      help='Version of the model to test (e.g., v1, v2)')
    parser.add_argument('--episodes', type=int, default=1,
                      help='Number of episodes to run')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                      help='Random seed')
    
    args = parser.parse_args()
    
    test_trained_model(
        model_type=args.model_type,
        version=args.version,
        episodes=args.episodes,
        seed=args.seed
    ) 