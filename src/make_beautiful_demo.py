"""
Beautiful Demo Video Generator
==============================
Creates high-quality demo videos comparing different algorithms.
"""

import numpy as np
import imageio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.grid_world import GridWorldEnv
from env.ma_wrapper import CentralizedWrapper
from agents.greedy import GreedyAgent
from agents.frontier_agent import FrontierAgent, CoordinatedFrontierAgent
from viz.beautiful_renderer import BeautifulRenderer
from stable_baselines3 import PPO


def run_frontier_demo(
    output_path: str = "frontier_demo.mp4",
    width: int = 50,
    height: int = 50,
    n_agents: int = 5,
    fov: int = 10,
    max_steps: int = 800,
    fps: int = 30,
    interp_frames: int = 2
):
    """Run frontier-based exploration demo."""
    print(f"Creating Frontier Agent demo...")
    
    env = GridWorldEnv(width, height, n_agents, fov_radius=fov, obs_mode='grid')
    renderer = BeautifulRenderer(width, height, cell_size=12)
    
    # Initialize agents
    CoordinatedFrontierAgent.reset_coordination()
    agents = {i: CoordinatedFrontierAgent(i, agent_repulsion=3.0) for i in range(n_agents)}
    
    frames = []
    obs, _ = env.reset(seed=42)
    renderer.reset_trails()
    
    for step in range(max_steps):
        # Get actions from frontier agents
        actions = {}
        for i in range(n_agents):
            actions[i] = agents[i].predict(obs[i])
        
        # Render interpolation frames
        for f in range(interp_frames):
            interp = (f + 1) / interp_frames
            frame = renderer.render(
                env,
                title="Frontier Exploration",
                fov_radius=fov,
                interpolation=interp if f < interp_frames - 1 else 1.0,
                show_trails=True,
                show_fov=True
            )
            frames.append(frame)
        
        # Step
        obs, _, _, truncated, _ = env.step(actions)
        
        # Check coverage
        coverage = np.sum(env.visited_map) / (width * height) * 100
        if step % 100 == 0:
            print(f"  Step {step}: Coverage {coverage:.1f}%")
        
        if truncated or coverage > 95:
            # Hold final frame
            for _ in range(fps * 2):
                frame = renderer.render(env, title="Frontier Exploration", fov_radius=fov)
                frames.append(frame)
            break
    
    env.close()
    
    print(f"Saving {len(frames)} frames to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Done! Final coverage: {coverage:.1f}%")
    return coverage


def run_rl_demo(
    model_path: str,
    output_path: str,
    title: str,
    width: int = 50,
    height: int = 50,
    n_agents: int = 5,
    fov: int = 5,
    obs_mode: str = 'grid',
    max_steps: int = 800,
    fps: int = 30,
    interp_frames: int = 2
):
    """Run RL agent demo."""
    print(f"Creating RL demo: {title}...")
    
    env_raw = GridWorldEnv(width, height, n_agents, fov_radius=fov, obs_mode=obs_mode)
    env = CentralizedWrapper(env_raw)
    renderer = BeautifulRenderer(width, height, cell_size=12)
    
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return 0
    
    frames = []
    obs, _ = env.reset(seed=42)
    renderer.reset_trails()
    
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=False)
        
        # Render interpolation frames
        for f in range(interp_frames):
            interp = (f + 1) / interp_frames
            frame = renderer.render(
                env.unwrapped,
                title=title,
                fov_radius=fov,
                interpolation=interp if f < interp_frames - 1 else 1.0,
                show_trails=True,
                show_fov=True
            )
            frames.append(frame)
        
        obs, _, _, truncated, _ = env.step(action)
        
        coverage = np.sum(env.unwrapped.visited_map) / (width * height) * 100
        if step % 100 == 0:
            print(f"  Step {step}: Coverage {coverage:.1f}%")
        
        if truncated or coverage > 95:
            for _ in range(fps * 2):
                frame = renderer.render(env.unwrapped, title=title, fov_radius=fov)
                frames.append(frame)
            break
    
    env.close()
    
    print(f"Saving {len(frames)} frames to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Done! Final coverage: {coverage:.1f}%")
    return coverage


def run_comparison_demo(
    output_path: str = "comparison_demo.mp4",
    width: int = 50,
    height: int = 50,
    n_agents: int = 5,
    max_steps: int = 600,
    fps: int = 30
):
    """
    Create a side-by-side comparison video.
    Shows Random, Greedy, Frontier, and RL agents.
    """
    print("Creating comparison demo...")
    
    # We'll render each algorithm sequentially with title cards
    all_frames = []
    
    scenarios = [
        ("Random Walk", "random", None, 5),
        ("Greedy Agent", "greedy", None, 10),
        ("Frontier Explorer", "frontier", None, 10),
    ]
    
    # Try to add RL if model exists
    if os.path.exists("models/ppo_grid_final.zip"):
        scenarios.append(("PPO (RL)", "rl", "models/ppo_grid_final", 5))
    
    for title, agent_type, model_path, fov in scenarios:
        print(f"\nRecording: {title}")
        
        # Title card
        renderer = BeautifulRenderer(width, height, cell_size=12)
        title_frame = np.zeros((renderer.window_height, renderer.window_width, 3), dtype=np.uint8)
        title_frame[:] = (30, 30, 46)  # Background color
        
        # Add title text (simple approach)
        for _ in range(fps * 2):  # 2 seconds
            all_frames.append(title_frame)
        
        # Run scenario
        env = GridWorldEnv(width, height, n_agents, fov_radius=fov, obs_mode='grid')
        
        if agent_type == "rl":
            env = CentralizedWrapper(env)
            model = PPO.load(model_path)
        
        obs, _ = env.reset(seed=42) if agent_type != "rl" else env.reset(seed=42)
        renderer.reset_trails()
        
        # Initialize agents
        if agent_type == "greedy":
            agents = {i: GreedyAgent(i, persistence_steps=20) for i in range(n_agents)}
        elif agent_type == "frontier":
            CoordinatedFrontierAgent.reset_coordination()
            agents = {i: CoordinatedFrontierAgent(i) for i in range(n_agents)}
        
        for step in range(max_steps):
            # Get actions
            if agent_type == "random":
                actions = {i: np.random.randint(0, 4) for i in range(n_agents)}
            elif agent_type == "greedy":
                actions = {i: agents[i].predict(obs[i]) for i in range(n_agents)}
            elif agent_type == "frontier":
                actions = {i: agents[i].predict(obs[i]) for i in range(n_agents)}
            elif agent_type == "rl":
                action, _ = model.predict(obs, deterministic=False)
                actions = action  # Already in correct format for wrapper
            
            # Render
            raw_env = env.unwrapped if agent_type == "rl" else env
            frame = renderer.render(
                raw_env,
                title=title,
                fov_radius=fov,
                show_trails=True,
                show_fov=agent_type != "random"
            )
            all_frames.append(frame)
            
            # Step
            if agent_type == "rl":
                obs, _, _, truncated, _ = env.step(actions)
            else:
                obs, _, _, truncated, _ = env.step(actions)
            
            coverage = np.sum(raw_env.visited_map) / (width * height) * 100
            
            if truncated or coverage > 90:
                # Hold final frame
                for _ in range(fps):
                    all_frames.append(frame)
                break
        
        if agent_type == "rl":
            env.close()
        else:
            env.close()
        
        print(f"  Final coverage: {coverage:.1f}%")
    
    print(f"\nSaving comparison video ({len(all_frames)} frames)...")
    imageio.mimsave(output_path, all_frames, fps=fps)
    print("Done!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate beautiful demo videos")
    parser.add_argument("--mode", choices=["frontier", "rl", "comparison", "all"], default="frontier")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--fps", type=int, default=30)
    
    args = parser.parse_args()
    
    if args.mode == "frontier" or args.mode == "all":
        output = args.output or "frontier_demo.mp4"
        run_frontier_demo(output, max_steps=args.steps, fps=args.fps)
    
    if args.mode == "rl" or args.mode == "all":
        output = args.output or "rl_demo.mp4"
        if os.path.exists("models/ppo_grid_final.zip"):
            run_rl_demo(
                "models/ppo_grid_final",
                output,
                "PPO Agent",
                max_steps=args.steps,
                fps=args.fps
            )
        else:
            print("RL model not found, skipping...")
    
    if args.mode == "comparison" or args.mode == "all":
        output = args.output or "comparison_demo.mp4"
        run_comparison_demo(output, max_steps=args.steps, fps=args.fps)


if __name__ == "__main__":
    main()

