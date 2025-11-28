import time
import numpy as np
from env.grid_world import GridWorldEnv

def main():
    # Initialize Environment
    # We use 'human' render mode to see the window
    env = GridWorldEnv(
        width=50, 
        height=50, 
        n_agents=5, 
        fov_radius=5, 
        obs_mode='grid', 
        render_mode='human'
    )

    # Reset
    obs, _ = env.reset()
    
    print("Starting Random Walk Simulation...")
    print(f"Grid Size: {env.width}x{env.height}")
    print(f"Agents: {env.n_agents}")
    
    try:
        for step in range(500):
            # Generate random actions for all agents
            # 0: Stay, 1: Up, 2: Down, 3: Left, 4: Right
            actions = {i: np.random.randint(0, 5) for i in range(env.n_agents)}
            
            # Step
            obs, rewards, dones, truncated, infos = env.step(actions)
            
            # Render
            env.render()
            
            # Simple logging
            if step % 50 == 0:
                total_reward = sum(rewards.values())
                print(f"Step {step}: Total Step Reward: {total_reward:.2f}")
                
            # Slow down slightly for visualization
            # time.sleep(0.05)
            
            if truncated:
                print("Episode finished (truncated).")
                break
                
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()

