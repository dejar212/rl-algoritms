import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from env.grid_world import GridWorldEnv
from env.ma_wrapper import CentralizedWrapper
from utils.custom_cnn import CustomCombinedExtractor
import os

def main():
    # Parameters
    width, height = 50, 50
    n_agents = 5
    fov = 5
    total_timesteps = 200_000
    
    # Create Environment (Target Mode)
    env = GridWorldEnv(width=width, height=height, n_agents=n_agents, fov_radius=fov, obs_mode='target')
    env = CentralizedWrapper(env)
    env = Monitor(env)
    
    # Policy Setup
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(cnn_output_dim=256),
        normalize_images=False 
    )
    
    print(f"Initializing PPO with Target Vector (MultiInputPolicy)...")
    
    model = PPO(
        "MultiInputPolicy", 
        env, 
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        tensorboard_log="./logs/target/",
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='./models_target/',
        name_prefix='ppo_target'
    )
    
    # Train
    print("Starting training...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    # Save Final
    model.save("models/ppo_target_final")
    print("Training finished. Model saved.")

if __name__ == "__main__":
    main()

