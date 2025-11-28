import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from env.grid_world_fov10 import GridWorldEnvFOV10
from env.ma_wrapper import CentralizedWrapper
from utils.custom_cnn import CustomCNN
import os

def main():
    # Parameters
    width, height = 50, 50
    n_agents = 5
    fov = 10 
    total_timesteps = 500_000 
    
    # Create Environment (FOV 10, Soft Rewards)
    env = GridWorldEnvFOV10(width=width, height=height, n_agents=n_agents, fov_radius=fov, obs_mode='grid')
    env = CentralizedWrapper(env)
    env = Monitor(env)
    
    # Policy Setup
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False 
    )
    
    print(f"Initializing PPO (FOV={fov}, Soft Rewards)...")
    
    model = PPO(
        "CnnPolicy", 
        env, 
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        tensorboard_log="./logs/fov10/",
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='./models_fov10/',
        name_prefix='ppo_fov10'
    )
    
    # Train
    print("Starting training...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    # Save Final
    model.save("models/ppo_fov10_final")
    print("Training finished. Model saved.")

if __name__ == "__main__":
    main()

