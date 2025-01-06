import numpy as np
import torch
from time import time
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from environment import ASVEnvironment
from config import setup_logger, DEFAULT_SEED, N_ENVS

# Setup logging
logger = setup_logger('ASV_Training')

# Check and set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.info("Using Metal Performance Shaders (MPS) device")
else:
    device = torch.device("cpu")
    logger.info("MPS not available, using CPU")

def make_env(seed=None):
    """Helper function to create an environment"""
    def _init():
        env = ASVEnvironment(seed=seed)
        return env
    return _init

def train_model(seed=DEFAULT_SEED):
    # Set seeds for all sources of randomness
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    # Create multiple environments for parallel training
    env = SubprocVecEnv([make_env(seed + i) for i in range(N_ENVS)])
    
    # Create the SAC model with seed
    model = SAC(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.002,
        gamma=0.99,
        ent_coef='auto',
        train_freq=4,
        gradient_steps=1,
        learning_starts=100_000,
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
        save_freq=100000,      # Increased from 50000
        save_path="./logs/",
        name_prefix="sac_asv"
    )
    
    # Train the model
    total_timesteps = 20_000_000
    logger.info(f"Starting training for {total_timesteps} timesteps...")

    start_time = time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    training_time = time() - start_time

    logger.info(f"Training completed in {training_time/60:.2f} minutes")
    logger.info(f"Average speed: {total_timesteps/training_time:.2f} timesteps/second")

    # Save the final model
    model.save("sac_asv_final")

if __name__ == "__main__":
    train_model()
