import numpy as np
import torch
import os
from time import time
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
import wandb

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


def make_env(seed=None):
    """Helper function to create an environment"""
    def _init():
        env = ASVEnvironment(seed=seed)
        return env
    return _init

def train_model(seed=DEFAULT_SEED):
    wandb.init(
        project="asv-navigation",
        name="sac_simplified_rewards",
        monitor_gym=True,
        sync_tensorboard=True,
        config={
            "model_type": "sac",
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
    
    # Get the wandb run path which includes timestamp
    run_path = wandb.run.dir
    
    # Set seeds for all sources of randomness
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    # Create multiple environments for parallel training
    env = SubprocVecEnv([make_env(seed + i) for i in range(N_ENVS)])
    

    tensorboard_path = os.path.join(run_path, "logs")
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
                pi=[256, 256, 128],
                qf=[256, 256, 128]
            )
        ),
        verbose=1,
        device=device,
        seed=seed,
        tensorboard_log=tensorboard_path,
    )
    
    # Train the model
    logger.info(f"Starting training for {TRAIN_TIMESTEPS} timesteps...")

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
    artifact.add_file(f"{model_save_path}")  # Stable-baselines adds .zip extension
    wandb.log_artifact(artifact)

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
