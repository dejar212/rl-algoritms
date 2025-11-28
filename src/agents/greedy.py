import numpy as np

class GreedyAgent:
    """
    A heuristic-based agent that moves towards the nearest unvisited cell 
    within its field of view. Uses persistence (inertia) to escape local optima.
    """
    def __init__(self, agent_id, persistence_steps=10):
        self.agent_id = agent_id
        self.persistence_steps = persistence_steps
        
        # State for persistence
        self.current_direction = 0
        self.steps_remaining = 0
        self.current_target = None # Stores (rel_x, rel_y) to target

    def predict(self, obs):
        """
        obs: np.array of shape (3, size, size)
             Channel 0: Obstacles (1=Wall)
             Channel 1: Agents (1=Agent)
             Channel 2: Visited (0=Unvisited, 1=Visited)
        """
        # Obs is (C, H, W)
        obstacles = obs[0]
        agents_map = obs[1]
        visited = obs[2]
        
        # Create a map of blocked cells (Walls + Other Agents)
        blocked = np.maximum(obstacles, agents_map)
        
        size = obstacles.shape[0]
        center = size // 2
        
        # 1. Look for Targets (Unvisited & Not Obstacle)
        candidates = np.argwhere((visited == 0) & (obstacles == 0))
        
        if len(candidates) > 0:
            # Found a target! Reset persistence and go for it.
            self.steps_remaining = 0
            self.current_direction = 0
            
            # Find nearest candidate
            distances = np.abs(candidates[:, 0] - center) + np.abs(candidates[:, 1] - center)
            nearest_idx = np.argmin(distances)
            target = candidates[nearest_idx]
            
            # Store target relative to agent for visualization
            # target index is (x, y) in local grid
            # relative vector is (x - center, y - center)
            self.current_target = (target[0] - center, target[1] - center)
            
            # Determine move towards target
            return self._move_towards(target, center, blocked)
        
        # 2. No visible target -> Use Persistence / Random Walk
        self.current_target = None
        return self._persistence_move(blocked, center)

    def _move_towards(self, target, center, blocked):
        tx, ty = target
        dx = tx - center
        dy = ty - center
        
        # Candidate actions sorted by preference (reducing distance)
        actions_to_try = []
        
        if abs(dx) >= abs(dy):
            actions_to_try.append(3 if dx < 0 else 4)
            if dy != 0: actions_to_try.append(1 if dy < 0 else 2)
        else:
            actions_to_try.append(1 if dy < 0 else 2)
            if dx != 0: actions_to_try.append(3 if dx < 0 else 4)
            
        # Try best actions first
        for action in actions_to_try:
            if self._is_valid_move(action, blocked, center):
                return action
                
        # If preferred moves blocked, try random valid move (stuck avoidance)
        return self._random_safe_move(blocked, center)

    def _persistence_move(self, blocked, center):
        # Check if current direction is still valid and active
        if self.steps_remaining > 0 and self.current_direction != 0:
            if self._is_valid_move(self.current_direction, blocked, center):
                self.steps_remaining -= 1
                return self.current_direction
        
        # Pick a new random direction
        valid_actions = self._get_valid_actions(blocked, center)
        if not valid_actions:
            return 0 # Stuck
            
        self.current_direction = np.random.choice(valid_actions)
        self.steps_remaining = self.persistence_steps
        return self.current_direction

    def _is_valid_move(self, action, blocked, center):
        if action == 1: # Up -> y-1
            return (center > 0 and blocked[center, center-1] == 0)
        elif action == 2: # Down -> y+1
            return (center < blocked.shape[1]-1 and blocked[center, center+1] == 0)
        elif action == 3: # Left -> x-1
            return (center > 0 and blocked[center-1, center] == 0)
        elif action == 4: # Right -> x+1
            return (center < blocked.shape[0]-1 and blocked[center+1, center] == 0)
        return False

    def _get_valid_actions(self, blocked, center):
        actions = []
        for a in [1, 2, 3, 4]:
            if self._is_valid_move(a, blocked, center):
                actions.append(a)
        return actions
        
    def _random_safe_move(self, blocked, center):
        actions = self._get_valid_actions(blocked, center)
        if actions:
            return np.random.choice(actions)
        return 0
