import time
import numpy as np
from stable_baselines3 import PPO
from env.grid_world import GridWorldEnv
from env.ma_wrapper import CentralizedWrapper
from utils.custom_cnn import CustomCNN
from collections import Counter

def main():
    # Use same config as training
    env_raw = GridWorldEnv(
        width=50, 
        height=50, 
        n_agents=5, 
        fov_radius=5, 
        obs_mode='grid', 
        render_mode='human'
    )
    env = CentralizedWrapper(env_raw)

    print("Loading model...")
    try:
        model = PPO.load("models/ppo_grid_final")
    except Exception as e:
        print(f"Standard load failed: {e}")
        return

    obs, _ = env.reset()
    
    print("Starting RL Agent Simulation...")
    
    total_cells = env_raw.width * env_raw.height
    
    try:
        for step in range(1000):
            # Try stochastic policy to see if deterministic was the issue
            action, _states = model.predict(obs, deterministic=False)
            
            # Log actions to see what's happening
            if step % 50 == 0:
                # Action is a numpy array of shape (5,)
                action_counts = Counter(action)
                print(f"Step {step} Actions: {dict(action_counts)}")
            
            obs, reward, terminated, truncated, info = env.step(action)
            env_raw.render()
            
            visited_count = np.sum(env_raw.visited_map)
            coverage_pct = (visited_count / total_cells) * 100
            
            if step % 50 == 0:
                print(f"Step {step}: Coverage {coverage_pct:.2f}% | Reward: {reward:.2f}")
                
            if truncated or coverage_pct >= 99.0:
                print(f"Finished at Step {step} with Coverage {coverage_pct:.2f}%")
                break
                
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
