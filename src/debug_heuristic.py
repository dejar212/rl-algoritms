import numpy as np
import gymnasium as gym
from env.grid_world import GridWorldEnv
import matplotlib.pyplot as plt

class HeuristicAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
    
    def predict(self, obs_dict):
        # obs_dict contains 'local' and 'target'
        # target is (dx, dy) normalized
        target_vec = obs_dict['target']
        
        if np.linalg.norm(target_vec) < 0.1:
            return np.random.randint(0, 5) # Random move if no target
            
        # Convert vector to action
        # 0: Stay, 1: Up, 2: Down, 3: Left, 4: Right
        # Vec is (dx, dy) in grid coords (x is width, y is height)
        # GridWorldEnv step:
        # 1: dy = -1 (Up)
        # 2: dy = 1 (Down)
        # 3: dx = -1 (Left)
        # 4: dx = 1 (Right)
        
        dx, dy = target_vec[0], target_vec[1]
        
        # Simple logic: pick dominant direction
        if abs(dx) > abs(dy):
            return 4 if dx > 0 else 3
        else:
            return 2 if dy > 0 else 1

def main():
    print("Debugging Heuristic Agent...")
    env = GridWorldEnv(width=50, height=50, n_agents=5, fov_radius=5, obs_mode='target')
    
    agents = [HeuristicAgent(i) for i in range(5)]
    
    obs, _ = env.reset(seed=42)
    
    total_cells = 50 * 50
    coverage_history = []
    
    for step in range(1000):
        actions = {}
        for i in range(5):
            actions[i] = agents[i].predict(obs[i])
            
        obs, _, _, _, _ = env.step(actions)
        
        visited_count = np.sum(env.visited_map)
        coverage_pct = (visited_count / total_cells) * 100
        coverage_history.append(coverage_pct)
        
        if step % 100 == 0:
            print(f"Step {step}: Coverage {coverage_pct:.1f}%")
            
    print(f"Final Coverage: {coverage_history[-1]:.1f}%")
    
    plt.plot(coverage_history)
    plt.title("Heuristic Agent (Follow Target Vector)")
    plt.ylabel("Coverage %")
    plt.xlabel("Steps")
    plt.savefig("debug_heuristic.png")
    print("Saved debug_heuristic.png")

if __name__ == "__main__":
    main()

