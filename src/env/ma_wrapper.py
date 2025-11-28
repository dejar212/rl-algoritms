import gymnasium as gym
import numpy as np

class CentralizedWrapper(gym.Wrapper):
    """
    Wraps the Multi-Agent GridWorld to appear as a Single-Agent environment
    for Centralized Training (one brain controls all agents).
    Supports both Box (single input) and Dict (multi-input) observation spaces.
    """
    def __init__(self, env):
        super().__init__(env)
        self.n_agents = env.n_agents
        # Action space: Vector of 4 actions for N agents (No Stay)
        self.action_space = gym.spaces.MultiDiscrete([4] * self.n_agents)
        
        # Observation Space Setup
        if isinstance(env.observation_space, gym.spaces.Dict):
            self.is_dict = True
            spaces_dict = {}
            for key, space in env.observation_space.spaces.items():
                if isinstance(space, gym.spaces.Box):
                    if len(space.shape) == 3:
                        new_shape = (space.shape[0] * self.n_agents, space.shape[1], space.shape[2])
                        spaces_dict[key] = gym.spaces.Box(low=space.low.min(), high=space.high.max(), shape=new_shape, dtype=space.dtype)
                    elif len(space.shape) == 1:
                        new_shape = (space.shape[0] * self.n_agents,)
                        spaces_dict[key] = gym.spaces.Box(low=space.low.min(), high=space.high.max(), shape=new_shape, dtype=space.dtype)
            
            self.observation_space = gym.spaces.Dict(spaces_dict)
            
        elif isinstance(env.observation_space, gym.spaces.Box):
            self.is_dict = False
            single_obs_shape = env.observation_space.shape
            if len(single_obs_shape) == 3:
                self.obs_h, self.obs_w = single_obs_shape[1], single_obs_shape[2]
                new_channels = single_obs_shape[0] * self.n_agents
                self.observation_space = gym.spaces.Box(
                    low=0, high=1, 
                    shape=(new_channels, self.obs_h, self.obs_w),
                    dtype=np.float32
                )
            elif len(single_obs_shape) == 1:
                new_dim = single_obs_shape[0] * self.n_agents
                self.observation_space = gym.spaces.Box(
                    low=-1, high=1, 
                    shape=(new_dim,),
                    dtype=np.float32
                )
        
    def reset(self, seed=None, options=None):
        obs_dict, info = self.env.reset(seed=seed, options=options)
        return self._flatten_obs(obs_dict), info
        
    def step(self, actions):
        # actions comes from SB3 as a numpy array [a1, a2, ..., aN]
        # Convert to dict {0: a1, 1: a2...}
        act_dict = {i: actions[i] for i in range(self.n_agents)}
        
        obs_dict, rewards, dones, truncated, infos = self.env.step(act_dict)
        
        # Aggregate rewards: We want the group to maximize total reward
        total_reward = sum(rewards.values())
        
        terminated = False # We rely on truncated
        
        return self._flatten_obs(obs_dict), total_reward, terminated, truncated, infos
        
    def _flatten_obs(self, obs_dict):
        if self.is_dict:
            keys = obs_dict[0].keys()
            flattened = {}
            for key in keys:
                item_list = [obs_dict[i][key] for i in range(self.n_agents)]
                flattened[key] = np.concatenate(item_list, axis=0)
            return flattened
        else:
            obs_list = [obs_dict[i] for i in range(self.n_agents)]
            stacked = np.concatenate(obs_list, axis=0)
            return stacked
