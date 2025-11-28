import os
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from env.grid_world import GridWorldEnv
from env.ma_wrapper import CentralizedWrapper
from utils.custom_cnn import CustomCNN

def make_env():
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
    log_dir = "logs/recurrent/"
    os.makedirs(log_dir, exist_ok=True)
    
    # 4 Parallel Environments
    n_envs = 4
    env = make_vec_env(make_env(), n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    env = VecMonitor(env, log_dir)

    # Custom Policy Config
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False,
        lstm_hidden_size=256,
        n_lstm_layers=1
    )

    model = RecurrentPPO(
        "CnnLstmPolicy", 
        env, 
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048, # Steps per env per update
        batch_size=64, # Micro-batch size for optimization
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path='./models_recurrent/',
        name_prefix='ppo_lstm'
    )

    print("Starting Recurrent PPO training (200k steps test)...")
    model.learn(
        total_timesteps=200_000, 
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    model.save("models/ppo_lstm_final")
    print("Training finished.")

if __name__ == "__main__":
    main()

