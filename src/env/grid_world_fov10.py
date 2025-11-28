import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

class GridWorldEnvFOV10(gym.Env):
    """
    Multi-Agent Grid World Environment (FOV 10, Soft Rewards, 4 Actions).
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, 
                 width=50, 
                 height=50, 
                 n_agents=5, 
                 fov_radius=10, 
                 obs_mode='grid', 
                 render_mode=None):
        
        super().__init__()
        self.width = width
        self.height = height
        self.n_agents = n_agents
        self.fov_radius = fov_radius
        self.obs_mode = obs_mode
        self.render_mode = render_mode
        
        # Action Space: 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space = spaces.Discrete(4)
        
        grid_size = 2 * self.fov_radius + 1
        
        if self.obs_mode == 'grid':
            self.observation_space = spaces.Box(
                low=0, high=1, 
                shape=(3, grid_size, grid_size), 
                dtype=np.float32
            )
        
        self.window = None
        self.clock = None
        self._reset_world()

    def _reset_world(self):
        self.grid = np.zeros((self.width, self.height), dtype=np.int8)
        self._generate_obstacles()
        self.visited_map = np.zeros((self.width, self.height), dtype=np.int8)
        self.agents = {}
        for i in range(self.n_agents):
            pos = self._get_random_free_pos()
            self.agents[i] = {
                "pos": np.array(pos),
                "active": True
            }
            self.visited_map[pos[0], pos[1]] = 1
        self.step_count = 0

    def _generate_obstacles(self):
        num_obstacles = int(self.width * self.height * 0.2)
        for _ in range(num_obstacles):
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            self.grid[x, y] = 1

    def _get_random_free_pos(self):
        while True:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            if self.grid[x, y] == 0:
                busy = False
                if hasattr(self, 'agents'):
                    for a in self.agents.values():
                        if np.array_equal(a['pos'], [x, y]):
                            busy = True
                            break
                if not busy:
                    return [x, y]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_world()
        return self._get_observations(), {}

    def step(self, actions):
        self.step_count += 1
        rewards = {i: 0.0 for i in range(self.n_agents)}
        dones = {i: False for i in range(self.n_agents)}
        infos = {i: {} for i in range(self.n_agents)}
        
        new_positions = {}
        for i, action in actions.items():
            rewards[i] -= 0.01 # Living cost
            agent = self.agents[i]
            curr_x, curr_y = agent['pos']
            dx, dy = 0, 0
            
            # Action mapping: 0:Up, 1:Down, 2:Left, 3:Right
            if action == 0: dy = -1    # Up
            elif action == 1: dy = 1   # Down
            elif action == 2: dx = -1  # Left
            elif action == 3: dx = 1   # Right
            
            target_x = np.clip(curr_x + dx, 0, self.width - 1)
            target_y = np.clip(curr_y + dy, 0, self.height - 1)
            new_positions[i] = np.array([target_x, target_y])

        # Collision: Obstacles
        for i, pos in new_positions.items():
            if self.grid[pos[0], pos[1]] == 1:
                rewards[i] -= 0.5 
                new_positions[i] = self.agents[i]['pos']

        # Collision: Agents
        occupied_cells = {}
        final_positions = {}
        for i, pos in new_positions.items():
            pos_tuple = tuple(pos)
            if pos_tuple in occupied_cells:
                rewards[i] -= 0.5
                final_positions[i] = self.agents[i]['pos']
            else:
                occupied_cells[pos_tuple] = i
                final_positions[i] = pos
        
        total_new_coverage = 0
        for i, pos in final_positions.items():
            self.agents[i]['pos'] = pos
            if self.visited_map[pos[0], pos[1]] == 0:
                rewards[i] += 1.0
                self.visited_map[pos[0], pos[1]] = 1
                total_new_coverage += 1

        for i in range(self.n_agents):
            rewards[i] += total_new_coverage * 0.1

        truncated = self.step_count >= 1000
        return self._get_observations(), rewards, dones, truncated, infos

    def _get_observations(self):
        obs = {}
        for i in range(self.n_agents):
            obs[i] = self._get_grid_obs(i)
        return obs

    def _get_grid_obs(self, agent_id):
        agent_pos = self.agents[agent_id]['pos']
        fov = self.fov_radius
        size = 2 * fov + 1
        local_grid = np.zeros((3, size, size), dtype=np.float32)
        
        min_x, max_x = agent_pos[0] - fov, agent_pos[0] + fov + 1
        min_y, max_y = agent_pos[1] - fov, agent_pos[1] + fov + 1
        
        start_x, start_y = max(0, min_x), max(0, min_y)
        offset_x, offset_y = start_x - min_x, start_y - min_y
        end_x, end_y = min(self.width, max_x), min(self.height, max_y)
        
        obs_slice = self.grid[start_x:end_x, start_y:end_y]
        local_grid[0, offset_x:offset_x+obs_slice.shape[0], offset_y:offset_y+obs_slice.shape[1]] = obs_slice
        
        for other_id, other_agent in self.agents.items():
            if other_id == agent_id: continue
            ox, oy = other_agent['pos']
            lx, ly = ox - min_x, oy - min_y
            if 0 <= lx < size and 0 <= ly < size:
                local_grid[1, lx, ly] = 1.0

        visited_slice = self.visited_map[start_x:end_x, start_y:end_y]
        local_grid[2, offset_x:offset_x+visited_slice.shape[0], offset_y:offset_y+visited_slice.shape[1]] = visited_slice
        
        if offset_x > 0: local_grid[0, :offset_x, :] = 1
        if offset_y > 0: local_grid[0, :, :offset_y] = 1
        if (size - (offset_x + obs_slice.shape[0])) > 0: local_grid[0, offset_x + obs_slice.shape[0]:, :] = 1
        return local_grid

    def close(self):
        if self.window is not None:
            pygame.quit()
