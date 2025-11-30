"""
Beautiful Grid World Renderer with Smooth Animations
=====================================================
Features:
- Smooth agent movement interpolation
- Trail visualization (agent paths)
- Gradient-based coverage heatmap
- Modern color palette
- Progress bar and stats overlay
"""

import numpy as np
import pygame
from typing import Dict, List, Tuple, Optional
from collections import deque
import math


class BeautifulRenderer:
    """
    High-quality renderer for multi-agent grid world visualization.
    Supports smooth animations, trails, and modern aesthetics.
    """
    
    # Modern color palette (Catppuccin Mocha inspired)
    COLORS = {
        'background': (30, 30, 46),      # Dark base
        'grid_line': (49, 50, 68),       # Subtle grid
        'wall': (88, 91, 112),           # Surface2
        'unvisited': (24, 24, 37),       # Mantle
        'visited_start': (166, 227, 161), # Green
        'visited_end': (137, 180, 250),   # Blue
        'agent_colors': [
            (243, 139, 168),  # Pink
            (250, 179, 135),  # Peach  
            (249, 226, 175),  # Yellow
            (166, 227, 161),  # Green
            (137, 180, 250),  # Blue
            (203, 166, 247),  # Mauve
        ],
        'trail_alpha': 80,
        'text': (205, 214, 244),         # Text
        'text_dim': (147, 153, 178),     # Subtext0
        'overlay_bg': (17, 17, 27, 220), # Crust with alpha
        'progress_bg': (49, 50, 68),
        'progress_fill': (166, 227, 161),
        'fov_fill': (137, 180, 250, 25),
        'fov_border': (137, 180, 250, 60),
    }
    
    def __init__(self, width: int, height: int, cell_size: int = 14):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        
        # Layout
        self.header_height = 60
        self.footer_height = 40
        self.margin = 20
        
        self.grid_width = width * cell_size
        self.grid_height = height * cell_size
        self.window_width = self.grid_width + 2 * self.margin
        self.window_height = self.grid_height + self.header_height + self.footer_height
        
        # Animation state
        self.prev_positions: Dict[int, np.ndarray] = {}
        self.agent_trails: Dict[int, deque] = {}
        self.trail_length = 30
        self.interpolation_progress = 1.0
        
        # Pygame setup
        self.surface = pygame.Surface((self.window_width, self.window_height))
        self.font_large = None
        self.font_medium = None
        self.font_small = None
        self._init_fonts()
        
    def _init_fonts(self):
        if not pygame.get_init():
            pygame.init()
        pygame.font.init()
        
        # Try to use a nice font, fallback to default
        try:
            self.font_large = pygame.font.SysFont("JetBrains Mono", 24, bold=True)
            self.font_medium = pygame.font.SysFont("JetBrains Mono", 16)
            self.font_small = pygame.font.SysFont("JetBrains Mono", 12)
        except:
            self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
            self.font_medium = pygame.font.SysFont("monospace", 16)
            self.font_small = pygame.font.SysFont("monospace", 12)
    
    def _lerp_color(self, c1: Tuple, c2: Tuple, t: float) -> Tuple:
        """Linear interpolation between two colors."""
        return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))
    
    def _get_coverage_color(self, visit_density: float) -> Tuple:
        """Get color based on how recently/often a cell was visited."""
        return self._lerp_color(
            self.COLORS['visited_start'],
            self.COLORS['visited_end'],
            min(1.0, visit_density)
        )
    
    def _grid_to_pixel(self, gx: float, gy: float) -> Tuple[float, float]:
        """Convert grid coordinates to pixel coordinates."""
        px = self.margin + gx * self.cell_size
        py = self.header_height + gy * self.cell_size
        return px, py
    
    def _draw_rounded_rect(self, surface, color, rect, radius=4):
        """Draw a rounded rectangle."""
        pygame.draw.rect(surface, color, rect, border_radius=radius)
    
    def _draw_header(self, title: str, step: int, coverage: float):
        """Draw the header with title and stats."""
        # Background
        header_rect = pygame.Rect(0, 0, self.window_width, self.header_height)
        pygame.draw.rect(self.surface, self.COLORS['background'], header_rect)
        
        # Title
        title_surf = self.font_large.render(title, True, self.COLORS['text'])
        self.surface.blit(title_surf, (self.margin, 15))
        
        # Stats on the right
        stats_text = f"Step: {step:04d}"
        stats_surf = self.font_medium.render(stats_text, True, self.COLORS['text_dim'])
        stats_rect = stats_surf.get_rect(topright=(self.window_width - self.margin, 20))
        self.surface.blit(stats_surf, stats_rect)
        
        # Separator line
        pygame.draw.line(
            self.surface,
            self.COLORS['grid_line'],
            (self.margin, self.header_height - 5),
            (self.window_width - self.margin, self.header_height - 5),
            2
        )
    
    def _draw_footer(self, coverage: float, n_agents: int):
        """Draw footer with progress bar."""
        footer_y = self.header_height + self.grid_height + 5
        
        # Progress bar background
        bar_width = self.grid_width - 100
        bar_height = 16
        bar_x = self.margin
        bar_y = footer_y + 10
        
        # Background
        pygame.draw.rect(
            self.surface,
            self.COLORS['progress_bg'],
            (bar_x, bar_y, bar_width, bar_height),
            border_radius=8
        )
        
        # Fill
        fill_width = int(bar_width * coverage / 100)
        if fill_width > 0:
            pygame.draw.rect(
                self.surface,
                self.COLORS['progress_fill'],
                (bar_x, bar_y, fill_width, bar_height),
                border_radius=8
            )
        
        # Percentage text
        pct_text = f"{coverage:.1f}%"
        pct_surf = self.font_medium.render(pct_text, True, self.COLORS['text'])
        pct_rect = pct_surf.get_rect(midleft=(bar_x + bar_width + 10, bar_y + bar_height // 2))
        self.surface.blit(pct_surf, pct_rect)
        
        # Agent count
        agent_text = f"Agents: {n_agents}"
        agent_surf = self.font_small.render(agent_text, True, self.COLORS['text_dim'])
        agent_rect = agent_surf.get_rect(topright=(self.window_width - self.margin, bar_y))
        self.surface.blit(agent_surf, agent_rect)
    
    def _draw_grid(self, grid: np.ndarray, visited: np.ndarray):
        """Draw the grid with walls and visited cells."""
        for x in range(self.width):
            for y in range(self.height):
                px, py = self._grid_to_pixel(x, y)
                rect = pygame.Rect(px, py, self.cell_size - 1, self.cell_size - 1)
                
                if grid[x, y] == 1:
                    # Wall
                    pygame.draw.rect(self.surface, self.COLORS['wall'], rect, border_radius=2)
                elif visited[x, y] == 1:
                    # Visited cell
                    color = self.COLORS['visited_start']
                    pygame.draw.rect(self.surface, color, rect, border_radius=2)
                else:
                    # Unvisited
                    pygame.draw.rect(self.surface, self.COLORS['unvisited'], rect, border_radius=2)
    
    def _draw_fov(self, agent_pos: np.ndarray, fov_radius: int, agent_color: Tuple):
        """Draw agent's field of view."""
        fov_size = (2 * fov_radius + 1) * self.cell_size
        fov_surf = pygame.Surface((fov_size, fov_size), pygame.SRCALPHA)
        
        # Fill with semi-transparent color
        fill_color = (*agent_color[:3], 20)
        fov_surf.fill(fill_color)
        
        # Border
        border_color = (*agent_color[:3], 50)
        pygame.draw.rect(fov_surf, border_color, fov_surf.get_rect(), 2, border_radius=4)
        
        # Position
        px, py = self._grid_to_pixel(agent_pos[0] - fov_radius, agent_pos[1] - fov_radius)
        self.surface.blit(fov_surf, (px, py))
    
    def _draw_trails(self, agents: Dict, interpolation: float = 1.0):
        """Draw agent movement trails."""
        for agent_id, agent in agents.items():
            if not agent['active']:
                continue
                
            if agent_id not in self.agent_trails:
                self.agent_trails[agent_id] = deque(maxlen=self.trail_length)
            
            # Get interpolated position
            current_pos = agent['pos']
            if agent_id in self.prev_positions and interpolation < 1.0:
                prev_pos = self.prev_positions[agent_id]
                interp_pos = prev_pos + (current_pos - prev_pos) * interpolation
            else:
                interp_pos = current_pos.astype(float)
            
            # Add to trail
            self.agent_trails[agent_id].append(interp_pos.copy())
            
            # Draw trail
            trail = list(self.agent_trails[agent_id])
            if len(trail) < 2:
                continue
                
            color = self.COLORS['agent_colors'][agent_id % len(self.COLORS['agent_colors'])]
            
            for i in range(len(trail) - 1):
                alpha = int(255 * (i / len(trail)) * 0.4)
                trail_color = (*color[:3], alpha)
                
                p1 = self._grid_to_pixel(trail[i][0] + 0.5, trail[i][1] + 0.5)
                p2 = self._grid_to_pixel(trail[i+1][0] + 0.5, trail[i+1][1] + 0.5)
                
                # Create a surface for alpha blending
                line_surf = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
                thickness = max(1, int(3 * (i / len(trail))))
                pygame.draw.line(line_surf, trail_color, p1, p2, thickness)
                self.surface.blit(line_surf, (0, 0))
    
    def _draw_agents(self, agents: Dict, interpolation: float = 1.0):
        """Draw agents with smooth interpolation."""
        for agent_id, agent in agents.items():
            if not agent['active']:
                continue
            
            color = self.COLORS['agent_colors'][agent_id % len(self.COLORS['agent_colors'])]
            
            # Interpolate position
            current_pos = agent['pos']
            if agent_id in self.prev_positions and interpolation < 1.0:
                prev_pos = self.prev_positions[agent_id]
                draw_pos = prev_pos + (current_pos - prev_pos) * interpolation
            else:
                draw_pos = current_pos.astype(float)
            
            # Convert to pixels
            px, py = self._grid_to_pixel(draw_pos[0] + 0.5, draw_pos[1] + 0.5)
            radius = int(self.cell_size * 0.4)
            
            # Glow effect
            glow_surf = pygame.Surface((radius * 6, radius * 6), pygame.SRCALPHA)
            for r in range(radius * 2, 0, -2):
                alpha = int(30 * (1 - r / (radius * 2)))
                glow_color = (*color[:3], alpha)
                pygame.draw.circle(glow_surf, glow_color, (radius * 3, radius * 3), r)
            self.surface.blit(glow_surf, (px - radius * 3, py - radius * 3))
            
            # Main circle with border
            pygame.draw.circle(self.surface, (255, 255, 255), (int(px), int(py)), radius + 2)
            pygame.draw.circle(self.surface, color, (int(px), int(py)), radius)
            
            # Agent ID
            id_surf = self.font_small.render(str(agent_id), True, (30, 30, 46))
            id_rect = id_surf.get_rect(center=(int(px), int(py)))
            self.surface.blit(id_surf, id_rect)
    
    def render(
        self,
        env,
        title: str = "Multi-Agent Patrol",
        fov_radius: Optional[int] = None,
        interpolation: float = 1.0,
        show_trails: bool = True,
        show_fov: bool = True
    ) -> np.ndarray:
        """
        Render the environment state.
        
        Args:
            env: GridWorldEnv instance
            title: Display title
            fov_radius: Field of view radius (optional)
            interpolation: Animation interpolation (0-1)
            show_trails: Whether to show agent trails
            show_fov: Whether to show field of view
            
        Returns:
            RGB frame as numpy array
        """
        # Clear background
        self.surface.fill(self.COLORS['background'])
        
        # Get env state
        grid = env.grid
        visited = env.visited_map
        agents = env.agents
        step = env.step_count
        
        # Calculate coverage
        total_cells = self.width * self.height
        obstacles = np.sum(grid)
        free_cells = total_cells - obstacles
        visited_count = np.sum(visited)
        coverage = (visited_count / free_cells) * 100 if free_cells > 0 else 0
        
        # Draw layers
        self._draw_grid(grid, visited)
        
        if show_fov and fov_radius:
            for agent_id, agent in agents.items():
                if agent['active']:
                    color = self.COLORS['agent_colors'][agent_id % len(self.COLORS['agent_colors'])]
                    self._draw_fov(agent['pos'], fov_radius, color)
        
        if show_trails:
            self._draw_trails(agents, interpolation)
        
        self._draw_agents(agents, interpolation)
        
        self._draw_header(title, step, coverage)
        self._draw_footer(coverage, len(agents))
        
        # Update previous positions for next frame
        if interpolation >= 1.0:
            self.prev_positions = {
                i: agent['pos'].copy() 
                for i, agent in agents.items()
            }
        
        # Convert to numpy array
        return np.transpose(pygame.surfarray.array3d(self.surface), (1, 0, 2))
    
    def reset_trails(self):
        """Clear all agent trails."""
        self.agent_trails.clear()
        self.prev_positions.clear()


def create_demo_video(
    env,
    agent_fn,
    output_path: str,
    title: str = "Demo",
    max_steps: int = 500,
    fps: int = 30,
    interpolation_frames: int = 3
):
    """
    Create a high-quality demo video with smooth animations.
    
    Args:
        env: Environment instance
        agent_fn: Function that takes obs and returns actions
        output_path: Output video path
        title: Video title
        max_steps: Maximum steps
        fps: Frames per second
        interpolation_frames: Frames between steps for smooth animation
    """
    import imageio
    
    renderer = BeautifulRenderer(env.width, env.height)
    frames = []
    
    obs, _ = env.reset(seed=42)
    renderer.reset_trails()
    
    for step in range(max_steps):
        # Get actions
        actions = agent_fn(obs)
        
        # Render interpolation frames
        for i in range(interpolation_frames):
            interp = (i + 1) / interpolation_frames
            frame = renderer.render(
                env,
                title=title,
                fov_radius=env.fov_radius,
                interpolation=interp if i < interpolation_frames - 1 else 1.0
            )
            frames.append(frame)
        
        # Step environment
        obs, _, _, truncated, _ = env.step(actions)
        
        if truncated or np.all(env.visited_map):
            # Add a few more frames at the end
            for _ in range(fps):
                frame = renderer.render(env, title=title, fov_radius=env.fov_radius)
                frames.append(frame)
            break
    
    print(f"Saving {len(frames)} frames to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Done! Video saved to {output_path}")

