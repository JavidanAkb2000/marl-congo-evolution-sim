"""
scripts/test_environment.py

Phase 2 visual verification script.

Renders the CongoEcosystem grid using Pygame so we can visually confirm:
    - The North Bank (scarce, scattered, low-energy food) behaves
      differently from the South Bank (abundant, clustered, high-energy
      food).
    - The Congo River barrier sits exactly at the correct row.
    - Food dynamically accumulates over time as step() is called each
      frame.

Run with:
    uv run scripts/test_environment.py
"""

from __future__ import annotations
import os
import sys

import pygame

# Ensure the project root is importable when this script is executed
# directly (e.g. `uv run scripts/test_environment.py`) regardless of the
# current working directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.environment.ecosystem import CongoEcosystem  # noqa: E402


# ----------------------------------------------------------------------
# Visual configuration constants
# ----------------------------------------------------------------------
GRID_WIDTH = 50
GRID_HEIGHT = 50
CELL_SIZE = 12  # pixels per grid cell

WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE
HUD_HEIGHT = 40  # extra vertical space reserved for on-screen stats

RIVER_THICKNESS_PX = 4  # visual thickness of the river divider line
TARGET_FPS = 8

# Color palette (RGB)
COLOR_NORTH_BG = (222, 184, 135)   # light brown / dry savanna
COLOR_SOUTH_BG = (152, 251, 152)   # light green / lush forest
COLOR_RIVER = (30, 144, 255)       # dodger blue
COLOR_NORTH_FOOD = (200, 30, 30)   # sparse red dots
COLOR_SOUTH_FOOD = (20, 160, 40)   # bright green clusters
COLOR_HUD_BG = (25, 25, 25)
COLOR_HUD_TEXT = (240, 240, 240)

NORTH_FOOD_RADIUS = 2
SOUTH_FOOD_RADIUS = 4


class EcosystemRenderer:
    """Handles all Pygame drawing for a CongoEcosystem instance."""

    def __init__(self, ecosystem: CongoEcosystem) -> None:
        self.ecosystem = ecosystem

        pygame.init()
        pygame.display.set_caption("Congo River Paradigm — Phase 2: Ecosystem & Resources")

        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT + HUD_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)

    def _draw_terrain(self) -> None:
        """Paint the North/South bank backgrounds and the river divider."""
        river_y = self.ecosystem.river_y

        # South Bank occupies rows [0, river_y] inclusive -> lush green.
        south_pixel_height = (river_y + 1) * CELL_SIZE
        south_rect = pygame.Rect(0, 0, WINDOW_WIDTH, south_pixel_height)
        pygame.draw.rect(self.screen, COLOR_SOUTH_BG, south_rect)

        # North Bank occupies rows [river_y + 1, height - 1] -> dry brown.
        north_pixel_top = south_pixel_height
        north_pixel_height = WINDOW_HEIGHT - south_pixel_height
        north_rect = pygame.Rect(0, north_pixel_top, WINDOW_WIDTH, north_pixel_height)
        pygame.draw.rect(self.screen, COLOR_NORTH_BG, north_rect)

        # River divider drawn centered on the boundary between the banks.
        river_rect = pygame.Rect(
            0,
            south_pixel_height - RIVER_THICKNESS_PX // 2,
            WINDOW_WIDTH,
            RIVER_THICKNESS_PX,
        )
        pygame.draw.rect(self.screen, COLOR_RIVER, river_rect)

    def _draw_food(self) -> None:
        """Draw every current food item, color-coded by territory type."""
        for item in self.ecosystem.food_items:
            center_x = item.x * CELL_SIZE + CELL_SIZE // 2
            center_y = item.y * CELL_SIZE + CELL_SIZE // 2

            if item.food_type == "north":
                pygame.draw.circle(
                    self.screen, COLOR_NORTH_FOOD, (center_x, center_y), NORTH_FOOD_RADIUS
                )
            else:
                pygame.draw.circle(
                    self.screen, COLOR_SOUTH_FOOD, (center_x, center_y), SOUTH_FOOD_RADIUS
                )

    def _draw_hud(self) -> None:
        """Draw a small stats bar beneath the grid."""
        hud_rect = pygame.Rect(0, WINDOW_HEIGHT, WINDOW_WIDTH, HUD_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_HUD_BG, hud_rect)

        north_count = len(self.ecosystem.get_food_by_type("north"))
        south_count = len(self.ecosystem.get_food_by_type("south"))

        stats_text = (
            f"Step: {self.ecosystem.current_step:5d}   "
            f"North Food: {north_count:4d}   "
            f"South Food: {south_count:4d}   "
            f"Total: {self.ecosystem.food_count():5d}   "
            f"(ESC or close window to quit)"
        )
        text_surface = self.font.render(stats_text, True, COLOR_HUD_TEXT)
        self.screen.blit(text_surface, (8, WINDOW_HEIGHT + 10))

    def render(self) -> None:
        """Draw a full frame: terrain, food, and HUD."""
        self._draw_terrain()
        self._draw_food()
        self._draw_hud()
        pygame.display.flip()

    def tick(self) -> None:
        """Advance the frame clock at the configured target FPS."""
        self.clock.tick(TARGET_FPS)

    def shutdown(self) -> None:
        """Cleanly tear down Pygame resources."""
        pygame.quit()


def run_simulation() -> None:
    """Main entry point: build the ecosystem and run the render loop."""
    ecosystem = CongoEcosystem(
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        river_y=25,
        north_spawn_prob=0.05,
        south_spawn_prob=0.50,
        north_food_energy=10.0,
        south_food_energy=40.0,
        south_cluster_size_range=(3, 6),
        south_cluster_radius=2,
    )
    ecosystem.reset()

    renderer = EcosystemRenderer(ecosystem)
    running = True

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            if not running:
                break

            ecosystem.step()
            renderer.render()
            renderer.tick()
    except KeyboardInterrupt:
        # Allow Ctrl+C in the terminal to exit gracefully as well.
        pass
    finally:
        renderer.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    run_simulation()