import numpy as np
import imageio
import gymnasium as gym
from stable_baselines3 import PPO
from env.grid_world_fov10 import GridWorldEnvFOV10
from env.ma_wrapper import CentralizedWrapper
from viz.renderer import GridWorldRenderer
import os
import sys
from utils.custom_cnn import CustomCNN

def record_validation(model_path, output_file="validation_fov10.mp4", max_steps=1000):
    print(f"Validating model: {model_path}")
    
    width, height = 50, 50
    n_agents = 5
    fov = 10
    
    # Create Env matching training config
    env_raw = GridWorldEnvFOV10(width=width, height=height, n_agents=n_agents, fov_radius=fov, obs_mode='grid')
    env = CentralizedWrapper(env_raw)
    
    try:
        # Load model
        # We assume CustomCNN is needed
        model = PPO.load(model_path, custom_objects={
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0
        })
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    renderer = GridWorldRenderer(width, height)
    frames = []
    
    obs, _ = env.reset(seed=123) # Fixed seed for consistency
    
    total_reward = 0
    coverage_history = []
    
    print("Starting simulation...")
    for step in range(max_steps):
        # Deterministic=False usually gives better exploration results for PPO unless fully converged
        action, _ = model.predict(obs, deterministic=False) 
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Calculate coverage
        coverage = np.mean(env_raw.visited_map) * 100
        coverage_history.append(coverage)
        
        # Render
        title = f"Validation FOV 10 | Step: {step} | Cov: {coverage:.1f}%"
        frame = renderer.render(env_raw, title=title, fov_radius=fov)
        frames.append(frame)
        
        if coverage >= 99.0:
            print(f"Full coverage reached at step {step}!")
            break
            
    env.close()
    
    final_cov = coverage_history[-1]
    print(f"Finished. Final Coverage: {final_cov:.1f}%. Total Reward: {total_reward:.1f}")
    
    print(f"Saving video to {output_file}...")
    imageio.mimsave(output_file, frames, fps=30, quality=8)
    print("Done!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Default to looking in models_downloaded
        # Find the latest zip file
        search_dir = "./models_downloaded"
        if os.path.exists(search_dir):
            files = [os.path.join(search_dir, f) for f in os.listdir(search_dir) if f.endswith('.zip')]
            if files:
                model_path = max(files, key=os.path.getmtime)
            else:
                model_path = "models/ppo_fov10_final.zip"
        else:
            model_path = "models/ppo_fov10_final.zip"
            
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Usage: python src/validate_fov10.py [path_to_model.zip]")
    else:
        record_validation(model_path)

