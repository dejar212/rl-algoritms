import numpy as np
import imageio
import gymnasium as gym
from stable_baselines3 import PPO
from env.grid_world import GridWorldEnv
from env.ma_wrapper import CentralizedWrapper
from agents.greedy import GreedyAgent
from viz.renderer import GridWorldRenderer
import os

def record_scenario(title, env, agent_type, model=None, max_steps=600):
    print(f"Recording scenario: {title}...")
    renderer = GridWorldRenderer(env.unwrapped.width, env.unwrapped.height)
    frames = []
    
    obs, _ = env.reset(seed=42)
    
    # Init Greedy Agents if needed
    greedy_agents = {}
    if agent_type == 'greedy':
        n_agents = env.unwrapped.n_agents
        greedy_agents = {i: GreedyAgent(i, persistence_steps=20) for i in range(n_agents)}
    
    for step in range(max_steps):
        actions = {}
        current_targets = {}
        
        if isinstance(env, CentralizedWrapper):
            # RL Mode
            if agent_type == 'rl':
                action, _ = model.predict(obs, deterministic=False)
                obs, _, _, _, _ = env.step(action)
        else:
            # Manual Mode
            if agent_type == 'random':
                 actions = {i: np.random.randint(0, 5) for i in range(env.n_agents)}
            elif agent_type == 'greedy':
                 for i in range(env.n_agents):
                     actions[i] = greedy_agents[i].predict(obs[i])
                     current_targets[i] = greedy_agents[i].current_target
            
            obs, _, _, _, _ = env.step(actions)
            
        # Render
        # Pass fov_radius from env
        frame = renderer.render(
            env.unwrapped, 
            title=title, 
            targets=current_targets, 
            fov_radius=env.unwrapped.fov_radius
        )
        frames.append(frame)
        
        # Early stop if 100%
        if np.all(env.unwrapped.visited_map):
            break
            
    return frames

def main():
    output_file = "patrol_demo_v2.mp4"
    all_frames = []
    
    width, height = 50, 50
    n_agents = 5
    
    # 1. Random Walk (Baseline)
    env_random = GridWorldEnv(width, height, n_agents, fov_radius=5, obs_mode='grid')
    frames_random = record_scenario("1. Random Walk (Blind)", env_random, 'random', max_steps=400)
    all_frames.extend(frames_random)
    env_random.close()
    
    # 2. RL PPO (Best RL so far)
    print("Loading PPO Model...")
    env_rl_raw = GridWorldEnv(width, height, n_agents, fov_radius=5, obs_mode='grid')
    env_rl = CentralizedWrapper(env_rl_raw)
    try:
        model = PPO.load("models/ppo_grid_final")
        frames_rl = record_scenario("2. RL PPO (FOV 5)", env_rl, 'rl', model=model, max_steps=600)
        all_frames.extend(frames_rl)
    except Exception as e:
        print(f"Skipping RL: {e}")
    env_rl.close()

    # 3. Greedy Agent (Target Performance)
    # Note: Using FOV 10 as requested for success showcase
    env_greedy = GridWorldEnv(width, height, n_agents, fov_radius=10, obs_mode='grid')
    frames_greedy = record_scenario("3. Greedy Agent (FOV 10 - Smart)", env_greedy, 'greedy', max_steps=600)
    all_frames.extend(frames_greedy)
    env_greedy.close()
    
    print(f"Saving video to {output_file} ({len(all_frames)} frames)...")
    imageio.mimsave(output_file, all_frames, fps=30, quality=8)
    print("Done!")

if __name__ == "__main__":
    main()
