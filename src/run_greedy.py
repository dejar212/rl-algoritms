import time
import numpy as np
from env.grid_world import GridWorldEnv
from agents.greedy import GreedyAgent
from collections import Counter

def main():
    # Reset FOV to 5 for baseline comparison
    env = GridWorldEnv(
        width=50, 
        height=50, 
        n_agents=5, 
        fov_radius=5, 
        obs_mode='grid', 
        render_mode='human'
    )

    obs, _ = env.reset()
    
    # Initialize Agents
    agents = {i: GreedyAgent(i, persistence_steps=20) for i in range(env.n_agents)}
    
    print("Starting Greedy Explorer Simulation...")
    
    total_cells = env.width * env.height
    
    try:
        for step in range(1000):
            actions = {}
            for i in range(env.n_agents):
                actions[i] = agents[i].predict(obs[i])
            
            # Debug: Action distribution
            if step % 50 == 0:
                action_counts = Counter(actions.values())
                print(f"Actions: {dict(action_counts)}")

            obs, rewards, dones, truncated, infos = env.step(actions)
            env.render()
            
            visited_count = np.sum(env.visited_map)
            coverage_pct = (visited_count / total_cells) * 100
            
            if step % 50 == 0:
                print(f"Step {step}: Coverage {coverage_pct:.2f}% | Reward: {sum(rewards.values()):.2f}")
                
            if truncated or coverage_pct >= 99.0:
                print(f"Finished at Step {step} with Coverage {coverage_pct:.2f}%")
                break
                
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
