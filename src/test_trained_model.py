from stable_baselines3 import SAC, PPO
import argparse
from pathlib import Path
import glob
from my_gymnasium.environment import ASVEnvironment
from config import setup_logger, DEFAULT_SEED
import matplotlib.pyplot as plt
import time
import os

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

def find_full_run_path(short_run_id):
    """Find the full run directory path from the short run ID"""
    # Get the project root directory (one level up from this script)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    possible_paths = glob.glob(os.path.join(project_root, f"wandb/run-*-{short_run_id}"))
    if not possible_paths:
        raise FileNotFoundError(f"No run directory found for run ID {short_run_id}")
    if len(possible_paths) > 1:
        logger.warning(f"Multiple run directories found for {short_run_id}, using most recent")
        # Sort by creation time and take the most recent
        return sorted(possible_paths, key=lambda x: Path(x).stat().st_ctime)[-1]
    return possible_paths[0]

def test_trained_model(model_type='sac', run_id=None, episodes=1, seed=DEFAULT_SEED):
    """Test the trained model for multiple episodes"""
    
    # Create environment with same seed
    env = ASVEnvironment(seed=seed)
    
    # Find the full run path from the short run ID
    try:
        full_run_path = find_full_run_path(run_id)
        logger.info(f"Found run directory: {full_run_path}")
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    # Look for the model file directly in the run directory
    model_path = Path(full_run_path) / "files" / "model.zip"
    if not model_path.exists():
        logger.error(f"Model file not found at expected path: {model_path}")
        return
    
    # Load the appropriate model type
    ModelClass = get_model_class(model_type)
    model = ModelClass.load(str(model_path))
    
    logger.info(f"Testing {model_type.upper()} model from run {run_id}")
    
    # Get git info from metadata
    metadata_path = Path(full_run_path) / "files" / "wandb-metadata.json"
    if metadata_path.exists():
        import json
        with open(metadata_path) as f:
            metadata = json.load(f)
            logger.info("Model trained with:")
            logger.info(f"Git commit: {metadata.get('git', {}).get('commit')}")
            logger.info(f"Branch: {metadata.get('git', {}).get('branch')}")
    
    for episode in range(episodes):
        logger.info(f"\nStarting Episode {episode + 1}")
        
        obs, _ = env.reset()
        # Log initial goal position
        goal_x, goal_y = env.goal
        logger.info(f"Goal position: ({goal_x:.2f}, {goal_y:.2f})")
        
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
            logger.info(
                f"Step {step}: "
                f"Action: ({action[0]:.3f}, {action[1]:.3f}), "
                f"Position: ({x:.2f}, {y:.2f}), "
                f"Reward: {reward:.2f}"
            )
            env.render()
            time.sleep(0.05)
            step += 1
            
        logger.info(f"Episode {episode + 1} finished after {step} steps. Total reward: {episode_reward:.2f}")
    
    # Keep the plot window open until manually closed
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a trained RL model')
    parser.add_argument('--model-type', type=str, default='sac',
                      help='Type of model to test (sac, ppo)')
    parser.add_argument('--run-id', type=str, required=True,
                      help='Short Wandb run ID (e.g., 1d9n4yxh)')
    parser.add_argument('--episodes', type=int, default=1,
                      help='Number of episodes to run')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                      help='Random seed')
    
    args = parser.parse_args()
    
    test_trained_model(
        model_type=args.model_type,
        run_id=args.run_id,
        episodes=args.episodes,
        seed=args.seed
    ) 