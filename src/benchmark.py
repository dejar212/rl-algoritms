import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from env.grid_world import GridWorldEnv
from env.ma_wrapper import CentralizedWrapper
from agents.greedy import GreedyAgent
from utils.custom_cnn import CustomCombinedExtractor, CustomCNN

def run_episode(env, agent_type, model=None, steps=1000):
    obs, _ = env.reset(seed=42) 
    total_cells = env.unwrapped.width * env.unwrapped.height
    coverage_history = []
    
    greedy_agents = {}
    if agent_type == 'greedy':
        greedy_agents = {i: GreedyAgent(i, persistence_steps=20) for i in range(env.unwrapped.n_agents)}
    
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    
    for step in range(steps):
        actions = {}
        
        if isinstance(env, CentralizedWrapper):
            if agent_type in ['rl_ppo', 'rl_mixed', 'rl_global']:
                action, _ = model.predict(obs, deterministic=False)
                obs, _, _, _, _ = env.step(action)
            elif agent_type == 'rl_lstm':
                action, lstm_states = model.predict(
                    obs, 
                    state=lstm_states, 
                    episode_start=episode_starts,
                    deterministic=False
                )
                obs, _, done, _, _ = env.step(action)
                episode_starts = np.array([done])
                
        else:
            if agent_type == 'random':
                 actions = {i: np.random.randint(0, 5) for i in range(env.n_agents)}
            elif agent_type == 'greedy':
                 for i in range(env.n_agents):
                     actions[i] = greedy_agents[i].predict(obs[i])
            obs, _, _, _, _ = env.step(actions)

        raw_env = env.unwrapped
        visited_count = np.sum(raw_env.visited_map)
        coverage_pct = (visited_count / total_cells) * 100
        coverage_history.append(coverage_pct)
        
    return coverage_history

def main():
    steps = 1000
    width, height = 50, 50
    n_agents = 5
    fov = 5
    
    results = {}
    print("Running Benchmark Phase 2.2...")
    
    # 1. Greedy
    print("Evaluating Greedy...")
    env_greedy = GridWorldEnv(width, height, n_agents, fov, obs_mode='grid')
    results['Greedy'] = run_episode(env_greedy, 'greedy', steps=steps)
    env_greedy.close()
    
    # 2. PPO (Image Only)
    print("Evaluating PPO (Image)...")
    env_ppo = CentralizedWrapper(GridWorldEnv(width, height, n_agents, fov, obs_mode='grid'))
    try:
        model_ppo = PPO.load("models/ppo_grid_final")
        results['PPO (Image)'] = run_episode(env_ppo, 'rl_ppo', model=model_ppo, steps=steps)
    except Exception as e: print(e)
    env_ppo.close()
    
    # 3. Global Map PPO
    print("Evaluating Global Map PPO...")
    env_global = CentralizedWrapper(GridWorldEnv(width, height, n_agents, fov, obs_mode='global'))
    try:
        model_global = PPO.load("models/ppo_global_final", custom_objects={"features_extractor_class": CustomCombinedExtractor})
        results['PPO (Global Map)'] = run_episode(env_global, 'rl_global', model=model_global, steps=steps)
    except Exception as e: print(f"Global Load Error: {e}")
    env_global.close()
    
    # Plot
    plt.figure(figsize=(12, 7))
    for name, data in results.items():
        if not data: continue
        plt.plot(data, label=f"{name}: {data[-1]:.1f}%", linewidth=2)
        
    plt.xlabel("Steps")
    plt.ylabel("Coverage (%)")
    plt.title(f"Benchmark: Global Map Integration")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("benchmark_global.png")
    print("Saved benchmark_global.png")

if __name__ == "__main__":
    main()
