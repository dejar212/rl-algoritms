import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from env.grid_world import GridWorldEnv
from env.ma_wrapper import CentralizedWrapper
from utils.custom_cnn import CustomCNN

def make_env():
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = GridWorldEnv(
            width=50, 
            height=50, 
            n_agents=5, 
            fov_radius=5, 
            obs_mode='grid'
        )
        env = CentralizedWrapper(env)
        return env
    return _init

def main():
    # Create log dir
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Parallel environments
    # We use 4 parallel envs to speed up data collection
    n_envs = 4
    env = make_vec_env(make_env(), n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    # Note: SubprocVecEnv is better for heavy envs. DummyVecEnv for simple ones.
    # GridWorld is simple python, but PPO needs lots of samples.
    
    env = VecMonitor(env, log_dir)

    # Define PPO model
    # We use CnnPolicy because input is 3D grid (Channels, H, W)
    # Custom CNN for small grid size
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False
    )

    model = PPO(
        "CnnPolicy", 
        env, 
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.01, # Entropy coefficient to encourage exploration
        policy_kwargs=policy_kwargs
    )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path='./models/',
        name_prefix='ppo_grid_centralized'
    )

    print("Starting PPO training...")
    model.learn(
        total_timesteps=200000, 
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    model.save("models/ppo_grid_final")
    print("Training finished. Model saved.")

if __name__ == "__main__":
    main()

