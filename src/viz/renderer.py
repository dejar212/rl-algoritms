import pygame
import numpy as np

class GridWorldRenderer:
    def __init__(self, width, height, cell_size=12):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.header_height = 40
        
        self.window_width = width * cell_size
        self.window_height = height * cell_size + self.header_height
        
        self.surface = pygame.Surface((self.window_width, self.window_height))
        self.font = None
        
        # Colors
        self.COLOR_BG = (250, 250, 250)
        self.COLOR_WALL = (60, 63, 65)
        self.COLOR_VISITED = (165, 214, 167) 
        self.COLOR_AGENT = (33, 150, 243)
        self.COLOR_TEXT = (50, 50, 50)
        self.COLOR_TARGET = (255, 82, 82) # Red for targets
        
    def render(self, env, title="Simulation", targets=None, fov_radius=None):
        """
        Renders the environment state.
        targets: dict {agent_id: (rel_x, rel_y)}
        fov_radius: int
        """
        if self.font is None:
            if not pygame.get_init():
                pygame.init()
            pygame.font.init()
            self.font = pygame.font.SysFont("Arial", 20, bold=True)
            
        self.surface.fill(self.COLOR_BG)
        
        # Draw Header
        pygame.draw.rect(self.surface, (230, 230, 230), (0, 0, self.window_width, self.header_height))
        pygame.draw.line(self.surface, (200, 200, 200), (0, self.header_height), (self.window_width, self.header_height), 2)
        
        grid = env.grid
        visited = env.visited_map
        agents = env.agents
        
        # 1. Draw Map
        for x in range(self.width):
            for y in range(self.height):
                rect = (x * self.cell_size, y * self.cell_size + self.header_height, self.cell_size, self.cell_size)
                if grid[x, y] == 1:
                    pygame.draw.rect(self.surface, self.COLOR_WALL, rect)
                elif visited[x, y] == 1:
                    pygame.draw.rect(self.surface, self.COLOR_VISITED, rect)
        
        # 2. Draw FOV (Under agents)
        if fov_radius:
            for i, agent in agents.items():
                if agent['active']:
                    self._draw_fov(agent['pos'], fov_radius)

        # 3. Draw Targets (Under agents)
        if targets:
            for i, target in targets.items():
                if target and agents[i]['active']:
                    self._draw_target(agents[i]['pos'], target)

        # 4. Draw Agents
        for i, agent in agents.items():
            if not agent['active']: continue
            ax, ay = agent['pos']
            
            center_x = int((ax + 0.5) * self.cell_size)
            center_y = int((ay + 0.5) * self.cell_size) + self.header_height
            radius = int(self.cell_size * 0.4)
            
            pygame.draw.circle(self.surface, (255, 255, 255), (center_x, center_y), radius + 2)
            pygame.draw.circle(self.surface, self.COLOR_AGENT, (center_x, center_y), radius)
            
        # 5. Draw Text
        total_cells = self.width * self.height
        visited_count = np.sum(visited)
        coverage_pct = (visited_count / total_cells) * 100
        step = env.step_count
        
        text_str = f"{title} | Step: {step} | Coverage: {coverage_pct:.1f}%"
        text_surf = self.font.render(text_str, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(midleft=(15, self.header_height // 2))
        self.surface.blit(text_surf, text_rect)
        
        return np.transpose(pygame.surfarray.array3d(self.surface), (1, 0, 2))

    def _draw_fov(self, agent_pos, fov):
        fov_px = (2 * fov + 1) * self.cell_size
        fov_surf = pygame.Surface((fov_px, fov_px), pygame.SRCALPHA)
        fov_surf.fill((33, 150, 243, 30)) # Light Blue, transparent
        pygame.draw.rect(fov_surf, (33, 150, 243, 80), fov_surf.get_rect(), 1)
        
        # Calculate top-left position (considering header)
        # Grid coordinates: agent_pos[0] - fov
        grid_x = agent_pos[0] - fov
        grid_y = agent_pos[1] - fov
        
        pixel_x = grid_x * self.cell_size
        pixel_y = grid_y * self.cell_size + self.header_height
        
        self.surface.blit(fov_surf, (pixel_x, pixel_y))

    def _draw_target(self, agent_pos, rel_target):
        # rel_target is (dx, dy)
        start_x = (agent_pos[0] + 0.5) * self.cell_size
        start_y = (agent_pos[1] + 0.5) * self.cell_size + self.header_height
        
        end_gx = agent_pos[0] + rel_target[0]
        end_gy = agent_pos[1] + rel_target[1]
        
        end_x = (end_gx + 0.5) * self.cell_size
        end_y = (end_gy + 0.5) * self.cell_size + self.header_height
        
        # Draw line
        pygame.draw.line(self.surface, self.COLOR_TARGET, (start_x, start_y), (end_x, end_y), 2)
        
        # Draw Marker (small circle)
        pygame.draw.circle(self.surface, self.COLOR_TARGET, (int(end_x), int(end_y)), 3)
