import numpy as np
import torch
import os
import argparse
from time import time
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
import wandb

from my_gymnasium.environment_base import ASVBaseEnvironment
from my_gymnasium.environment_obs import ASVObsEnvironment
from config import (
    setup_logger, 
    DEFAULT_SEED, 
    N_ENVS
)

# Setup logging
logger = setup_logger('SAC_Training')

TRAIN_TIMESTEPS = 5_000_000
START_LEARNING = 10_000

def make_env(seed=None, env_type='base'):
    """Helper function to create an environment
    
    Args:
        seed (int): Random seed for the environment
        env_type (str): Type of environment to create ('base' or 'obs')
    """
    def _init():
        if env_type == 'base':
            env = ASVBaseEnvironment(seed=seed)
        elif env_type == 'obs':
            env = ASVObsEnvironment(seed=seed)
        else:
            raise ValueError(f"Unknown environment type: {env_type}. Must be 'base' or 'obs'")
        return env
    return _init

def train_model(seed=DEFAULT_SEED, env_type='base'):
    wandb.init(
        project="asv-navigation",
        name=f"sac_{env_type}_env",
        monitor_gym=True,
        sync_tensorboard=True,
        config={
            "env_type": env_type,
            "train_timesteps": TRAIN_TIMESTEPS,
            "learning_starts": START_LEARNING,
        }
    )
    
    # Get the wandb run path which includes timestamp
    run_path = wandb.run.dir
    
    # Set seeds for all sources of randomness
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    # Create multiple environments for parallel training
    env = SubprocVecEnv([make_env(seed + i, env_type) for i in range(N_ENVS)])
    
    tensorboard_path = os.path.join(run_path, "logs")
    # Create the SAC model
    model = SAC(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=int(1e6),
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef='auto',
        train_freq=4,
        gradient_steps=1,
        learning_starts=10000,
        verbose=1,
        device=device,
        seed=seed,
        tensorboard_log=tensorboard_path,
    )
    
    # Train the model
    logger.info(f"Starting training for {TRAIN_TIMESTEPS} timesteps with {env_type} environment...")

    start_time = time()
    model.learn(
        total_timesteps=TRAIN_TIMESTEPS,
        progress_bar=True
    )
    training_time = time() - start_time

    logger.info(f"Training completed in {training_time/60:.2f} minutes")
    logger.info(f"Average speed: {TRAIN_TIMESTEPS/training_time:.2f} timesteps/second")

    # Save the final model
    model_save_path = os.path.join(run_path, "model.zip")
    model.save(model_save_path)
    logger.info(f"Model saved to {model_save_path}")

    # Log the model as a wandb artifact
    artifact = wandb.Artifact(f'sac_model_{wandb.run.id}', type='model')
    artifact.add_file(f"{model_save_path}")
    wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an SAC agent for ASV navigation')
    parser.add_argument('--env-type', type=str, choices=['base', 'obs'], default='base',
                      help='Type of environment to use (base: no obstacles, obs: with obstacles)')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                      help='Random seed')
    
    args = parser.parse_args()
    
    # Check and set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Metal Performance Shaders (MPS) device")
    else:
        device = torch.device("cpu")
        logger.info("MPS not available, using CPU")
        
    train_model(seed=args.seed, env_type=args.env_type)
