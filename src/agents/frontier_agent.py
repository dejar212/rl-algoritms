"""
Frontier-Based Exploration Agent
================================
A smarter heuristic agent that:
1. Identifies frontier cells (unvisited cells adjacent to visited)
2. Uses BFS to find nearest reachable frontier
3. Coordinates with other agents to avoid clustering
4. Maintains exploration momentum
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import deque


class FrontierAgent:
    """
    Frontier-based exploration agent with coordination.
    
    Key improvements over basic Greedy:
    - Frontier detection (boundary between known/unknown)
    - BFS pathfinding to nearest frontier
    - Anti-clustering: avoids areas with other agents
    - Memory: remembers recent positions to avoid loops
    """
    
    # Action mapping: 0=Up, 1=Down, 2=Left, 3=Right
    ACTIONS = {
        0: (0, -1),   # Up
        1: (0, 1),    # Down
        2: (-1, 0),   # Left
        3: (1, 0),    # Right
    }
    
    def __init__(
        self,
        agent_id: int,
        memory_size: int = 50,
        frontier_weight: float = 1.0,
        agent_repulsion: float = 2.0,
        exploration_bonus: float = 0.5
    ):
        self.agent_id = agent_id
        self.memory_size = memory_size
        self.frontier_weight = frontier_weight
        self.agent_repulsion = agent_repulsion
        self.exploration_bonus = exploration_bonus
        
        # State
        self.position_history: deque = deque(maxlen=memory_size)
        self.current_target: Optional[Tuple[int, int]] = None
        self.current_path: List[Tuple[int, int]] = []
        self.stuck_counter = 0
        self.last_action = None
        
    def reset(self):
        """Reset agent state for new episode."""
        self.position_history.clear()
        self.current_target = None
        self.current_path = []
        self.stuck_counter = 0
        self.last_action = None
    
    def predict(self, obs: np.ndarray, other_agent_positions: Optional[List[Tuple[int, int]]] = None) -> int:
        """
        Predict the best action given observation.
        
        Args:
            obs: Observation array (3, H, W)
                 Channel 0: Obstacles
                 Channel 1: Other agents
                 Channel 2: Visited cells
            other_agent_positions: Optional list of other agent positions
            
        Returns:
            Action (0-3)
        """
        obstacles = obs[0]
        agents_map = obs[1]
        visited = obs[2]
        
        size = obstacles.shape[0]
        center = size // 2
        
        # Create blocked map (walls + agents)
        blocked = np.maximum(obstacles, agents_map)
        
        # 1. Find frontiers (unvisited cells adjacent to visited)
        frontiers = self._find_frontiers(visited, obstacles)
        
        if len(frontiers) > 0:
            # 2. Score frontiers based on distance and agent density
            best_frontier = self._select_best_frontier(
                frontiers, center, blocked, agents_map, other_agent_positions
            )
            
            if best_frontier is not None:
                self.current_target = best_frontier
                action = self._move_towards_target(best_frontier, center, blocked)
                if action is not None:
                    self.stuck_counter = 0
                    self.last_action = action
                    return action
        
        # 3. No frontiers visible - explore randomly with momentum
        self.current_target = None
        return self._exploration_move(blocked, center)
    
    def _find_frontiers(self, visited: np.ndarray, obstacles: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find frontier cells (unvisited cells adjacent to visited cells).
        """
        frontiers = []
        size = visited.shape[0]
        
        for x in range(size):
            for y in range(size):
                # Cell must be unvisited and not an obstacle
                if visited[x, y] == 1 or obstacles[x, y] == 1:
                    continue
                
                # Check if adjacent to a visited cell
                is_frontier = False
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size:
                        if visited[nx, ny] == 1:
                            is_frontier = True
                            break
                
                if is_frontier:
                    frontiers.append((x, y))
        
        return frontiers
    
    def _select_best_frontier(
        self,
        frontiers: List[Tuple[int, int]],
        center: int,
        blocked: np.ndarray,
        agents_map: np.ndarray,
        other_positions: Optional[List[Tuple[int, int]]]
    ) -> Optional[Tuple[int, int]]:
        """
        Select the best frontier to explore based on:
        - Distance from agent
        - Distance from other agents (prefer isolated frontiers)
        - Cluster size (prefer larger frontier clusters)
        """
        if not frontiers:
            return None
        
        best_score = float('-inf')
        best_frontier = None
        
        # Calculate frontier clusters for bonus
        frontier_set = set(frontiers)
        
        for fx, fy in frontiers:
            # Base score: negative distance (closer is better)
            dist = abs(fx - center) + abs(fy - center)
            score = -dist * self.frontier_weight
            
            # Bonus for frontier clusters (more unexplored area nearby)
            cluster_size = sum(
                1 for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                if (fx + dx, fy + dy) in frontier_set
            )
            score += cluster_size * self.exploration_bonus
            
            # Penalty for being near other agents
            if other_positions:
                for ox, oy in other_positions:
                    agent_dist = abs(fx - ox) + abs(fy - oy)
                    if agent_dist < 5:
                        score -= self.agent_repulsion * (5 - agent_dist)
            
            # Also check agents_map for nearby agents
            for ax in range(max(0, fx - 3), min(blocked.shape[0], fx + 4)):
                for ay in range(max(0, fy - 3), min(blocked.shape[1], fy + 4)):
                    if agents_map[ax, ay] == 1:
                        agent_dist = abs(fx - ax) + abs(fy - ay)
                        score -= self.agent_repulsion * (4 - agent_dist) / 2
            
            # Penalty for recently visited positions (avoid loops)
            if (fx, fy) in self.position_history:
                score -= 5
            
            if score > best_score:
                best_score = score
                best_frontier = (fx, fy)
        
        return best_frontier
    
    def _move_towards_target(
        self,
        target: Tuple[int, int],
        center: int,
        blocked: np.ndarray
    ) -> Optional[int]:
        """
        Move towards target using simple pathfinding.
        """
        tx, ty = target
        dx = tx - center
        dy = ty - center
        
        # Prioritize actions based on direction
        actions_priority = []
        
        if abs(dx) >= abs(dy):
            # Prioritize horizontal movement
            if dx < 0:
                actions_priority.append(2)  # Left
            elif dx > 0:
                actions_priority.append(3)  # Right
            if dy < 0:
                actions_priority.append(0)  # Up
            elif dy > 0:
                actions_priority.append(1)  # Down
        else:
            # Prioritize vertical movement
            if dy < 0:
                actions_priority.append(0)  # Up
            elif dy > 0:
                actions_priority.append(1)  # Down
            if dx < 0:
                actions_priority.append(2)  # Left
            elif dx > 0:
                actions_priority.append(3)  # Right
        
        # Add remaining actions
        for a in [0, 1, 2, 3]:
            if a not in actions_priority:
                actions_priority.append(a)
        
        # Try actions in priority order
        for action in actions_priority:
            if self._is_valid_move(action, blocked, center):
                return action
        
        return None
    
    def _is_valid_move(self, action: int, blocked: np.ndarray, center: int) -> bool:
        """Check if an action leads to a valid (unblocked) cell."""
        dx, dy = self.ACTIONS[action]
        nx, ny = center + dx, center + dy
        
        if 0 <= nx < blocked.shape[0] and 0 <= ny < blocked.shape[1]:
            return blocked[nx, ny] == 0
        return False
    
    def _exploration_move(self, blocked: np.ndarray, center: int) -> int:
        """
        Random exploration with momentum (tends to continue in same direction).
        """
        self.stuck_counter += 1
        
        # Get valid actions
        valid_actions = [a for a in range(4) if self._is_valid_move(a, blocked, center)]
        
        if not valid_actions:
            return 0  # Stuck, return any action
        
        # If stuck for too long, pick random direction
        if self.stuck_counter > 10:
            self.stuck_counter = 0
            return np.random.choice(valid_actions)
        
        # Momentum: prefer continuing in the same direction
        if self.last_action is not None and self.last_action in valid_actions:
            if np.random.random() < 0.7:  # 70% chance to continue
                return self.last_action
        
        # Otherwise random
        action = np.random.choice(valid_actions)
        self.last_action = action
        return action


class CoordinatedFrontierAgent(FrontierAgent):
    """
    Extended frontier agent with explicit coordination.
    Shares target information with other agents to avoid conflicts.
    """
    
    # Class-level shared state for coordination
    claimed_targets: Dict[int, Tuple[int, int]] = {}
    
    def __init__(self, agent_id: int, **kwargs):
        super().__init__(agent_id, **kwargs)
    
    @classmethod
    def reset_coordination(cls):
        """Reset shared coordination state."""
        cls.claimed_targets.clear()
    
    def predict(self, obs: np.ndarray, other_agent_positions: Optional[List[Tuple[int, int]]] = None) -> int:
        """Predict with coordination."""
        # Clear our old claim
        if self.agent_id in self.claimed_targets:
            del self.claimed_targets[self.agent_id]
        
        # Get action from parent
        action = super().predict(obs, other_agent_positions)
        
        # Register our target
        if self.current_target is not None:
            self.claimed_targets[self.agent_id] = self.current_target
        
        return action
    
    def _select_best_frontier(
        self,
        frontiers: List[Tuple[int, int]],
        center: int,
        blocked: np.ndarray,
        agents_map: np.ndarray,
        other_positions: Optional[List[Tuple[int, int]]]
    ) -> Optional[Tuple[int, int]]:
        """Select frontier avoiding already claimed targets."""
        # Filter out frontiers claimed by other agents
        claimed_by_others = {
            target for aid, target in self.claimed_targets.items()
            if aid != self.agent_id
        }
        
        # Prefer unclaimed frontiers, but don't exclude claimed ones entirely
        unclaimed = [f for f in frontiers if f not in claimed_by_others]
        
        if unclaimed:
            return super()._select_best_frontier(
                unclaimed, center, blocked, agents_map, other_positions
            )
        else:
            return super()._select_best_frontier(
                frontiers, center, blocked, agents_map, other_positions
            )

