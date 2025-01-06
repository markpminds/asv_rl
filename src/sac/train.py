import numpy as np
import torch
from time import time
import re
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback

from my_gymnasium.environment import ASVEnvironment
from config import (
    setup_logger, 
    MAX_SPEED, 
    MAX_ANGULAR_VELOCITY, 
    TURN_RATE_FACTOR, 
    DEFAULT_SEED, 
    N_ENVS
)

# Setup logging
logger = setup_logger('SAC_Training')

TRAIN_TIMESTEPS = 1_000_000

def get_next_version() -> int:
    """Get the next version number by checking existing SAC model directories"""
    model_dir = Path("models") / "sac"
    
    if not model_dir.exists():
        # First version if directory doesn't exist
        model_dir.mkdir(parents=True)
        return 1
        
    # List all version directories
    version_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('v')]
    
    if not version_dirs:
        return 1
        
    # Extract version numbers and find max
    version_numbers = []
    for d in version_dirs:
        match = re.match(r'v(\d+)', d.name)
        if match:
            version_numbers.append(int(match.group(1)))
            
    return max(version_numbers, default=0) + 1

def make_env(seed=None):
    """Helper function to create an environment"""
    def _init():
        env = ASVEnvironment(seed=seed)
        return env
    return _init

def train_model(seed=DEFAULT_SEED):
    # Get next version number
    version = get_next_version()
    model_path = Path("models") / "sac" / f"v{version}"
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project="asv-navigation",
        name=f"sac_v{version}",
        config={
            "model_type": "sac",
            "version": version,
            "training_timesteps": TRAIN_TIMESTEPS,
            "env_type": "simplified_rewards",
            "reward_structure": {
                "heading_weight": 2.0,
                "progress_weight": 3.0,
                "success_reward": 100.0,
                "out_of_bounds_penalty": -25.0,
            },
            "model_config": {
                "learning_rate": 3e-4,
                "batch_size": 256,
                "buffer_size": TRAIN_TIMESTEPS,
                "train_freq": 4,
                "learning_starts": int(TRAIN_TIMESTEPS/20)
            },
            "env_config": {
                "max_speed": MAX_SPEED,
                "max_angular_velocity": MAX_ANGULAR_VELOCITY,
                "turn_rate_factor": TURN_RATE_FACTOR
            }
        }
    )
    
    # Set seeds for all sources of randomness
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    # Create multiple environments for parallel training
    env = SubprocVecEnv([make_env(seed + i) for i in range(N_ENVS)])
    
    # Create the SAC model
    model = SAC(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=TRAIN_TIMESTEPS,
        batch_size=256,
        tau=0.002,
        gamma=0.99,
        ent_coef='auto',
        train_freq=4,
        gradient_steps=1,
        learning_starts=int(TRAIN_TIMESTEPS/20),
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],
                qf=[256, 256]
            ),
            log_std_init=-3,
        ),
        verbose=1,
        device=device,
        seed=seed
    )
    
    # Setup checkpointing
    checkpoint_callback = CheckpointCallback(
        save_freq=int(TRAIN_TIMESTEPS/10),
        save_path=str(model_path),
        name_prefix="sac_asv"
    )
    
    wandb_callback = WandbCallback(
        gradient_save_freq=int(TRAIN_TIMESTEPS/10),
        model_save_path=str(model_path),
        verbose=2,
    )
    
    # Train the model
    logger.info(f"Starting training for {TRAIN_TIMESTEPS} timesteps...")

    start_time = time()
    model.learn(
        total_timesteps=TRAIN_TIMESTEPS,
        callback=[checkpoint_callback, wandb_callback],
        progress_bar=True
    )
    training_time = time() - start_time

    logger.info(f"Training completed in {training_time/60:.2f} minutes")
    logger.info(f"Average speed: {TRAIN_TIMESTEPS/training_time:.2f} timesteps/second")

    # Save final model
    final_model_path = model_path / "sac_asv"
    model.save(str(final_model_path))
    wandb.save(str(final_model_path) + ".zip")
    wandb.finish()

if __name__ == "__main__":
    # Check and set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Metal Performance Shaders (MPS) device")
    else:
        device = torch.device("cpu")
        logger.info("MPS not available, using CPU")
        
    train_model()

