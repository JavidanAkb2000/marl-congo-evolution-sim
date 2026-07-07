"""
scripts/visualize_simulation.py

Phase 5: The Pygame Visualization GUI Layer.

Wraps the Phase 4 CongoArena simulation engine in a real-time, split-
screen Pygame window:
    - Left panel: the live 50x50 grid, terrain-shaded by bank, with
      agents colored by a continuous gene-based gradient and food
      rendered as glowing gold circles.
    - Right panel: a HUD showing live population statistics, a color
      legend, and on-screen controls help.
    - Interaction flashes: combat/attack events flash a red cross over
      the affected cell; births flash a soft pink heart marker.
    - Keyboard controls: SPACE pauses/resumes, UP/DOWN adjust
      simulation speed, S saves an immediate checkpoint, ESC quits.

This module does not reimplement any simulation logic — it reuses
`build_arena()` from `scripts/run_simulation.py` so the exact same
ecological calibration (spawn rates, energy costs, mutation rates,
etc.) that the CLI verification script uses is guaranteed to be
identical here. It drives the engine purely through
`CongoArena.step()` and `CheckpointManager.save()`, so all Phase 4
invariants (gene-sum normalization, hard grid boundaries, checkpoint
round-trip fidelity) are preserved untouched.

Run with:
    uv run scripts/visualize_simulation.py
"""

from __future__ import annotations

import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import pygame

# Ensure the project root is importable when this script is executed
# directly (e.g. `uv run scripts/visualize_simulation.py`) regardless of
# the current working directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.environment.arena import CongoArena  # noqa: E402
from src.persistence.checkpoint import CheckpointManager  # noqa: E402
from run_simulation import build_arena  # noqa: E402

# ----------------------------------------------------------------------
# Window & grid layout constants
# ----------------------------------------------------------------------
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

GRID_COLS = 50
GRID_ROWS = 50
CELL_SIZE = 12  # pixels per grid cell -> 600x600 grid panel (compact, fits 1080p safely)

GRID_MARGIN_X = 24
GRID_MARGIN_Y = 24
GRID_ORIGIN = (GRID_MARGIN_X, GRID_MARGIN_Y)
GRID_PIXEL_WIDTH = GRID_COLS * CELL_SIZE
GRID_PIXEL_HEIGHT = GRID_ROWS * CELL_SIZE

HUD_ORIGIN_X = GRID_ORIGIN[0] + GRID_PIXEL_WIDTH + 32
HUD_WIDTH = WINDOW_WIDTH - HUD_ORIGIN_X - 24

RIVER_THICKNESS_PX = 4

# ----------------------------------------------------------------------
# Color palette
# ----------------------------------------------------------------------
COLOR_BG = (18, 18, 22)
COLOR_NORTH_BG = (222, 184, 135)   # dry savanna
COLOR_SOUTH_BG = (152, 251, 152)   # lush forest
COLOR_RIVER = (30, 144, 255)

COLOR_FOOD_CORE = (255, 250, 210)
COLOR_FOOD_GLOW = (255, 215, 0)
# Food decay/rot color stages: fresh (yellow) -> aging (orange) -> rotting (red).
COLOR_FOOD_FRESH = (255, 215, 0)     # yellow-gold, just spawned
COLOR_FOOD_AGING = (255, 140, 0)     # orange, roughly half-aged
COLOR_FOOD_ROTTING = (220, 40, 40)   # red, about to disappear

# Gorilla territory: a dark, massive "impassable" block on the reserved
# hotspots. Semi-transparent fill so rotting food still shows THROUGH it
# (the whole point — chimps can't reach it, so it visibly rots on top of
# the gorilla mass), plus a light-grey outline so the block reads clearly
# against both the tan savanna background and the red of rotting food.
COLOR_GORILLA_FILL = (35, 35, 40)          # near-black dark grey (RGB base)
COLOR_GORILLA_FILL_ALPHA = 200             # mostly opaque, but lets food peek through
COLOR_GORILLA_OUTLINE = (150, 150, 160)    # light grey border for clear separation
COLOR_GORILLA_ICON = (15, 15, 18)          # even darker centre "core" mass

# Gene-based agent color gradient anchors.
GENE_COLOR_COOPERATIVE = (0, 200, 120)   # G_T -> 0.0 : vibrant emerald green
GENE_COLOR_HYBRID = (255, 165, 0)        # G_T -> 0.5 : orange
GENE_COLOR_AGGRESSIVE = (220, 20, 60)    # G_T -> 1.0 : crimson red
AGENT_OUTLINE_COLOR = (20, 20, 20)

COLOR_HUD_BG = (26, 27, 33)
COLOR_HUD_PANEL_BORDER = (60, 62, 70)
COLOR_HUD_TITLE = (245, 245, 250)
COLOR_HUD_TEXT = (210, 212, 220)
COLOR_HUD_ACCENT = (120, 200, 255)
COLOR_HUD_WARN = (255, 200, 90)
COLOR_HUD_PAUSED = (255, 90, 90)
COLOR_HUD_SAVED = (140, 255, 170)

COLOR_COMBAT_FLASH = (255, 50, 50)
COLOR_BIRTH_FLASH = (255, 182, 220)

# ----------------------------------------------------------------------
# Flash timings (milliseconds)
# ----------------------------------------------------------------------
COMBAT_FLASH_DURATION_MS = 450
BIRTH_FLASH_DURATION_MS = 900

# ----------------------------------------------------------------------
# Simulation speed bounds
# ----------------------------------------------------------------------
DEFAULT_FPS = 10
MIN_FPS = 1
MAX_FPS = 60
FPS_STEP = 2

DEFAULT_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "congo_checkpoint.json")


# ----------------------------------------------------------------------
# Gene -> color interpolation
# ----------------------------------------------------------------------
def _lerp_color(
    color_a: Tuple[int, int, int], color_b: Tuple[int, int, int], t: float
) -> Tuple[int, int, int]:
    """Linearly interpolate between two RGB colors at fraction t in [0, 1]."""
    t = max(0.0, min(1.0, t))
    return tuple(int(round(color_a[i] + (color_b[i] - color_a[i]) * t)) for i in range(3))


def gene_to_color(g_t: float) -> Tuple[int, int, int]:
    """Map an agent's G_T (aggression) gene to a continuous RGB gradient.

    G_T = 0.0 -> vibrant emerald green (pure cooperative)
    G_T = 0.5 -> orange (balanced / hybrid, mutated drift)
    G_T = 1.0 -> crimson red (pure aggressive)

    The gradient is piecewise-linear through the orange midpoint so
    that near-0.5 genotypes read clearly as "hybrid" rather than being
    washed out by a single-segment blend.
    """
    g_t = max(0.0, min(1.0, g_t))
    if g_t <= 0.5:
        return _lerp_color(GENE_COLOR_COOPERATIVE, GENE_COLOR_HYBRID, g_t / 0.5)
    return _lerp_color(GENE_COLOR_HYBRID, GENE_COLOR_AGGRESSIVE, (g_t - 0.5) / 0.5)


# ----------------------------------------------------------------------
# Event log parsing -> flash triggers
# ----------------------------------------------------------------------
_COMBAT_PATTERN = re.compile(r"^COMBAT: Agent (\d+) .*? vs Agent (\d+) .*? -> winner: Agent (\d+)")
_ATTACK_PATTERN = re.compile(r"^ATTACK: Agent (\d+) preys on Agent (\d+)")
_REPRODUCTION_PATTERN = re.compile(
    r"^REPRODUCTION: Agent (\d+) x Agent (\d+) -> offspring Agent (\d+)"
)


class FlashManager:
    """Tracks short-lived visual markers (combat/birth flashes) over time.

    Positions are recorded in *grid* coordinates (not pixels); the
    renderer converts to pixel space when drawing. Flash lifetime is
    tracked in real wall-clock milliseconds via `pygame.time.get_ticks()`
    so flash duration stays consistent regardless of the current
    simulation speed (FPS).
    """

    def __init__(self) -> None:
        self._flashes: List[dict] = []

    def add(self, x: int, y: int, kind: str, duration_ms: int) -> None:
        self._flashes.append(
            {"x": x, "y": y, "kind": kind, "start": pygame.time.get_ticks(), "duration": duration_ms}
        )

    def active_flashes(self) -> List[Tuple[int, int, str, int]]:
        """Return (x, y, kind, alpha) for every still-active flash, and
        prune expired ones internally as a side effect."""
        now = pygame.time.get_ticks()
        alive: List[dict] = []
        result: List[Tuple[int, int, str, int]] = []

        for flash in self._flashes:
            elapsed = now - flash["start"]
            if elapsed <= flash["duration"]:
                fraction_remaining = 1.0 - (elapsed / flash["duration"])
                alpha = max(0, min(255, int(255 * fraction_remaining)))
                result.append((flash["x"], flash["y"], flash["kind"], alpha))
                alive.append(flash)

        self._flashes = alive
        return result

    def process_step_log(
        self, log_lines: List[str], positions_before_step: Dict[int, Tuple[int, int]], arena: CongoArena
    ) -> None:
        """Scan one step's event log and register the appropriate flashes.

        Combat/attack flashes use each participant's position *before*
        this step's interaction phase ran (captured by the caller prior
        to `arena.step()`), since a defeated agent may already have been
        pruned from `arena.agents_by_id` by the time we inspect the log.
        Birth flashes use the newborn's actual (guaranteed-alive)
        post-step position.
        """
        for line in log_lines:
            combat_match = _COMBAT_PATTERN.match(line)
            attack_match = _ATTACK_PATTERN.match(line)
            reproduction_match = _REPRODUCTION_PATTERN.match(line)

            if combat_match:
                participant_id = int(combat_match.group(1))
                position = positions_before_step.get(participant_id)
                if position is not None:
                    self.add(position[0], position[1], "combat", COMBAT_FLASH_DURATION_MS)

            elif attack_match:
                attacker_id = int(attack_match.group(1))
                position = positions_before_step.get(attacker_id)
                if position is not None:
                    self.add(position[0], position[1], "combat", COMBAT_FLASH_DURATION_MS)

            elif reproduction_match:
                offspring_id = int(reproduction_match.group(3))
                offspring = arena.agents_by_id.get(offspring_id)
                if offspring is not None and offspring.is_alive:
                    self.add(offspring.x, offspring.y, "birth", BIRTH_FLASH_DURATION_MS)


# ----------------------------------------------------------------------
# Renderer
# ----------------------------------------------------------------------
class SimulationRenderer:
    """Handles all Pygame drawing for the split-screen simulation view."""

    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen

        self.font_title = pygame.font.SysFont("consolas", 22, bold=True)
        self.font_section = pygame.font.SysFont("consolas", 16, bold=True)
        self.font_stat = pygame.font.SysFont("consolas", 16)
        self.font_small = pygame.font.SysFont("consolas", 13)

        self._food_glow_surface = self._build_food_glow_surface(radius=5)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    #
    # IMPORTANT — Y-AXIS ORIENTATION FIX:
    # `arena.ecosystem.get_bank(y)` defines South Bank as y <= river_y
    # (the numerically LOW rows) and North Bank as y > river_y (the
    # numerically HIGH rows). Pygame's screen-space Y axis increases
    # downward, so naively mapping grid_y -> pixel_y unflipped would put
    # South (low y) at the TOP of the window and North (high y) at the
    # BOTTOM — geographically backwards from the expected "North is up"
    # map convention.
    #
    # We fix this ONCE, here, by flipping the row index before
    # converting to pixels. Every other drawing routine (terrain, food,
    # agents, flashes) calls this same method, so flipping it in this
    # single place keeps the whole scene perfectly in sync — there is no
    # way for an agent/food item to be computed against one Y mapping
    # while the terrain uses another.
    @staticmethod
    def _flipped_row(grid_y: int, grid_height: int) -> int:
        """Invert a grid row so higher logical Y renders nearer the top."""
        return (grid_height - 1) - grid_y

    def _cell_center_px(self, grid_x: int, grid_y: int, grid_height: int) -> Tuple[int, int]:
        flipped_y = self._flipped_row(grid_y, grid_height)
        px = GRID_ORIGIN[0] + grid_x * CELL_SIZE + CELL_SIZE // 2
        py = GRID_ORIGIN[1] + flipped_y * CELL_SIZE + CELL_SIZE // 2
        return px, py

    # ------------------------------------------------------------------
    # Precomputed assets
    # ------------------------------------------------------------------
    @staticmethod
    def _build_food_glow_surface(radius: int) -> pygame.Surface:
        """Build a small radial-gradient sprite for a 'glowing' food item."""
        size = radius * 2 + 8
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        center = (size // 2, size // 2)

        for r in range(radius + 4, 0, -1):
            falloff = (1.0 - r / (radius + 4)) ** 2
            alpha = int(200 * falloff)
            pygame.draw.circle(surface, (*COLOR_FOOD_GLOW, alpha), center, r)

        pygame.draw.circle(surface, (*COLOR_FOOD_CORE, 255), center, max(2, radius - 2))
        return surface

    # ------------------------------------------------------------------
    # Left panel: terrain, food, agents, flashes
    # ------------------------------------------------------------------
    def _draw_terrain(self, arena: CongoArena) -> None:
        river_y = arena.ecosystem.river_y
        grid_height = arena.ecosystem.height

        # North Bank = rows (river_y, height-1] -> after the Y-flip these
        # are the rows nearest the TOP of the screen, so North is painted
        # first, starting at the grid's pixel origin.
        north_row_count = grid_height - river_y - 1
        north_pixel_height = north_row_count * CELL_SIZE
        north_rect = pygame.Rect(
            GRID_ORIGIN[0], GRID_ORIGIN[1], GRID_PIXEL_WIDTH, north_pixel_height
        )
        pygame.draw.rect(self.screen, COLOR_NORTH_BG, north_rect)

        # South Bank = rows [0, river_y] -> after the Y-flip these are the
        # rows nearest the BOTTOM of the screen, so South fills the
        # remainder of the grid panel beneath the North block.
        south_pixel_top = GRID_ORIGIN[1] + north_pixel_height
        south_pixel_height = GRID_PIXEL_HEIGHT - north_pixel_height
        south_rect = pygame.Rect(
            GRID_ORIGIN[0], south_pixel_top, GRID_PIXEL_WIDTH, south_pixel_height
        )
        pygame.draw.rect(self.screen, COLOR_SOUTH_BG, south_rect)

        # The river divider sits exactly on the boundary between the two
        # painted blocks, i.e. exactly where the flipped coordinate
        # system places the river_y/river_y+1 boundary.
        river_rect = pygame.Rect(
            GRID_ORIGIN[0],
            south_pixel_top - RIVER_THICKNESS_PX // 2,
            GRID_PIXEL_WIDTH,
            RIVER_THICKNESS_PX,
        )
        pygame.draw.rect(self.screen, COLOR_RIVER, river_rect)

        border_rect = pygame.Rect(GRID_ORIGIN[0], GRID_ORIGIN[1], GRID_PIXEL_WIDTH, GRID_PIXEL_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_HUD_PANEL_BORDER, border_rect, width=2)

    @staticmethod
    def _freshness_color(freshness: float) -> Tuple[int, int, int]:
        """Map freshness in [0,1] to a color: 1.0 yellow -> 0.5 orange -> 0.0 red.

        Piecewise-linear through the orange midpoint so the three decay
        stages (fresh / aging / rotting) each read clearly rather than
        blending into a muddy single gradient.
        """
        f = max(0.0, min(1.0, freshness))
        if f >= 0.5:
            # 1.0 -> 0.5 : fresh (yellow) to aging (orange)
            t = (1.0 - f) / 0.5
            a, b = COLOR_FOOD_FRESH, COLOR_FOOD_AGING
        else:
            # 0.5 -> 0.0 : aging (orange) to rotting (red)
            t = (0.5 - f) / 0.5
            a, b = COLOR_FOOD_AGING, COLOR_FOOD_ROTTING
        return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))

    def _draw_gorillas(self, arena: CongoArena) -> None:
        """Draw the translucent 'gorilla territory' zone over each reserved hotspot.

        Rendered AFTER terrain but BEFORE food, so rotting food still shows
        on top of the zone — visually telling the story: the food is right
        there, but the silverback troop sitting on it keeps the chimpanzees
        out, so it just rots. The solid gorilla "mass" core is drawn
        separately by `_draw_gorilla_cores`, AFTER food, so the animal
        itself is never buried under a pile of fruit sprites. The block
        spans the full hotspot radius (the zone chimps are displaced from),
        drawn via the same Y-flipped coordinate transform as everything
        else so it lands exactly on the logic's gorilla zone.
        """
        for rect in self._gorilla_zone_rects(arena):
            left, top, width, height = rect
            block = pygame.Surface((width, height), pygame.SRCALPHA)
            block.fill((*COLOR_GORILLA_FILL, COLOR_GORILLA_FILL_ALPHA))
            self.screen.blit(block, (left, top))
            pygame.draw.rect(
                self.screen, COLOR_GORILLA_OUTLINE, pygame.Rect(left, top, width, height), width=2
            )

    def _draw_gorilla_cores(self, arena: CongoArena) -> None:
        """Draw the solid dark gorilla 'mass' at each reserved hotspot center.

        Drawn AFTER food so the occupying troop always reads clearly as a
        heavy, present block even when fruit has piled up (and is rotting)
        around it inside the zone.
        """
        for rect in self._gorilla_zone_rects(arena):
            left, top, width, height = rect
            core = max(CELL_SIZE * 2, min(width, height) // 3)
            core_rect = pygame.Rect(0, 0, core, core)
            core_rect.center = (left + width // 2, top + height // 2)
            pygame.draw.rect(self.screen, COLOR_GORILLA_ICON, core_rect, border_radius=3)
            pygame.draw.rect(self.screen, COLOR_GORILLA_OUTLINE, core_rect, width=2, border_radius=3)

    def _gorilla_zone_rects(self, arena: CongoArena) -> List[Tuple[int, int, int, int]]:
        """Compute the clamped (left, top, width, height) pixel rect of each
        gorilla-occupied hotspot zone. Shared by the zone and core drawing
        passes so both stay perfectly aligned.
        """
        eco = arena.ecosystem
        grid_height = eco.height
        radius = eco.north_hotspot_radius
        rects: List[Tuple[int, int, int, int]] = []

        for index, (hx, hy) in enumerate(eco.north_hotspots):
            if not eco.hotspot_state[index]["gorilla_occupied"]:
                continue

            # The gorilla zone spans cells [hx-radius, hx+radius] x
            # [hy-radius, hy+radius]. Convert the two extreme cell centers
            # to pixel space (Y-flip makes the "top" pixel come from the
            # HIGHER grid-y), then build the covering rect from them.
            cx_min, cy_a = self._cell_center_px(hx - radius, hy - radius, grid_height)
            cx_max, cy_b = self._cell_center_px(hx + radius, hy + radius, grid_height)

            left = min(cx_min, cx_max) - CELL_SIZE // 2
            top = min(cy_a, cy_b) - CELL_SIZE // 2
            width = abs(cx_max - cx_min) + CELL_SIZE
            height = abs(cy_b - cy_a) + CELL_SIZE

            # Clamp to the grid panel so a hotspot near the edge doesn't
            # bleed the block outside the play area.
            left = max(GRID_ORIGIN[0], left)
            top = max(GRID_ORIGIN[1], top)
            right = min(GRID_ORIGIN[0] + GRID_PIXEL_WIDTH, left + width)
            bottom = min(GRID_ORIGIN[1] + GRID_PIXEL_HEIGHT, top + height)
            width = max(0, right - left)
            height = max(0, bottom - top)
            if width <= 0 or height <= 0:
                continue

            rects.append((left, top, width, height))
        return rects

    def _draw_food(self, arena: CongoArena) -> None:
        grid_height = arena.ecosystem.height
        # Food is drawn as small SQUARES (agents are circles), so the two
        # never blur together, and so food reads as little "pixels" nestled
        # inside the larger square hotspot/gorilla blocks. Side length is
        # kept below the cell size so individual items stay distinct even
        # when several pile up in one cell.
        side = max(3, CELL_SIZE - 4)
        half = side // 2
        for item in arena.ecosystem.food_items:
            cx, cy = self._cell_center_px(item.x, item.y, grid_height)
            freshness = arena.ecosystem.get_food_freshness(item)
            color = self._freshness_color(freshness)
            # A soft square halo (dimmer as it rots) plus a solid square
            # core, so fresh food still "glows" while rotting food reads as
            # a flat, dull red block about to vanish.
            halo_alpha = int(90 * freshness)
            if halo_alpha > 0:
                halo_side = side + 4
                halo = pygame.Surface((halo_side, halo_side), pygame.SRCALPHA)
                halo.fill((*color, halo_alpha))
                self.screen.blit(halo, (cx - halo_side // 2, cy - halo_side // 2))
            pygame.draw.rect(self.screen, color, pygame.Rect(cx - half, cy - half, side, side))

    def _draw_agents(self, arena: CongoArena) -> None:
        grid_height = arena.ecosystem.height
        radius = max(3, CELL_SIZE // 2 - 1)
        for agent in arena.agents_by_id.values():
            if not agent.is_alive:
                continue
            cx, cy = self._cell_center_px(agent.x, agent.y, grid_height)
            color = gene_to_color(agent.g_t)
            pygame.draw.circle(self.screen, color, (cx, cy), radius)
            pygame.draw.circle(self.screen, AGENT_OUTLINE_COLOR, (cx, cy), radius, width=1)

    def _draw_flashes(self, arena: CongoArena, flash_manager: FlashManager) -> None:
        grid_height = arena.ecosystem.height
        for grid_x, grid_y, kind, alpha in flash_manager.active_flashes():
            cx, cy = self._cell_center_px(grid_x, grid_y, grid_height)
            marker_size = CELL_SIZE * 2
            marker_surface = pygame.Surface((marker_size, marker_size), pygame.SRCALPHA)

            if kind == "combat":
                color = (*COLOR_COMBAT_FLASH, alpha)
                pad = 3
                pygame.draw.line(marker_surface, color, (pad, pad), (marker_size - pad, marker_size - pad), 3)
                pygame.draw.line(marker_surface, color, (marker_size - pad, pad), (pad, marker_size - pad), 3)
                pygame.draw.rect(
                    marker_surface, color, (1, 1, marker_size - 2, marker_size - 2), width=2
                )
            elif kind == "birth":
                color = (*COLOR_BIRTH_FLASH, alpha)
                mid_x, mid_y = marker_size // 2, marker_size // 2
                lobe_radius = marker_size // 5
                pygame.draw.circle(marker_surface, color, (mid_x - lobe_radius // 2, mid_y - lobe_radius // 2), lobe_radius)
                pygame.draw.circle(marker_surface, color, (mid_x + lobe_radius // 2, mid_y - lobe_radius // 2), lobe_radius)
                pygame.draw.polygon(
                    marker_surface,
                    color,
                    [
                        (mid_x - lobe_radius - lobe_radius // 2, mid_y - lobe_radius // 3),
                        (mid_x + lobe_radius + lobe_radius // 2, mid_y - lobe_radius // 3),
                        (mid_x, mid_y + lobe_radius * 2),
                    ],
                )

            self.screen.blit(marker_surface, (cx - marker_size // 2, cy - marker_size // 2))

    # ------------------------------------------------------------------
    # Right panel: HUD
    # ------------------------------------------------------------------
    def _draw_hud(
        self,
        arena: CongoArena,
        fps: int,
        paused: bool,
        save_message: str,
    ) -> None:
        panel_rect = pygame.Rect(HUD_ORIGIN_X - 12, GRID_ORIGIN[1] - 4, HUD_WIDTH + 12, GRID_PIXEL_HEIGHT + 8)
        pygame.draw.rect(self.screen, COLOR_HUD_BG, panel_rect, border_radius=8)
        pygame.draw.rect(self.screen, COLOR_HUD_PANEL_BORDER, panel_rect, width=2, border_radius=8)

        x = HUD_ORIGIN_X
        y = GRID_ORIGIN[1] + 8

        y = self._blit_line(x, y, "CONGO RIVER PARADIGM", self.font_title, COLOR_HUD_TITLE)
        y = self._blit_line(x, y, "Phase 5 — Live MARL Simulation", self.font_small, COLOR_HUD_ACCENT)
        y += 14

        # --- Living stats ---
        alive_agents = [a for a in arena.agents_by_id.values() if a.is_alive]
        population = len(alive_agents)
        north_count = sum(1 for a in alive_agents if arena.ecosystem.get_bank(a.y) == "north")
        south_count = population - north_count
        avg_g_t = (sum(a.g_t for a in alive_agents) / population) if population else 0.0
        food_count = arena.ecosystem.food_count()
        gorilla_zones = sum(1 for s in arena.ecosystem.hotspot_state if s["gorilla_occupied"])
        open_patches = sum(
            1
            for s in arena.ecosystem.hotspot_state
            if not s["gorilla_occupied"] and not s["depleted"]
        )

        y = self._blit_line(x, y, "LIVING STATS", self.font_section, COLOR_HUD_ACCENT)
        y = self._blit_stat(x, y, "Step", f"{arena.current_step}")
        y = self._blit_stat(x, y, "Total Population", f"{population}")
        y = self._blit_stat(x, y, "North Agents (Chimp)", f"{north_count}")
        y = self._blit_stat(x, y, "South Agents (Bonobo)", f"{south_count}")
        y = self._blit_stat(x, y, "Avg G_T (Aggression)", f"{avg_g_t:.3f}")
        y = self._blit_stat(x, y, "Food on Grid", f"{food_count}")
        y = self._blit_stat(x, y, "Gorilla Zones (blocked)", f"{gorilla_zones}")
        y = self._blit_stat(x, y, "Open North Patches", f"{open_patches}")
        y += 10

        # --- Legend ---
        y = self._blit_line(x, y, "LEGEND", self.font_section, COLOR_HUD_ACCENT)
        y = self._blit_swatch_line(x, y, GENE_COLOR_COOPERATIVE, "Cooperative  (G_T -> 0.0)")
        y = self._blit_swatch_line(x, y, GENE_COLOR_HYBRID, "Hybrid / Mutated (G_T ~ 0.5)")
        y = self._blit_swatch_line(x, y, GENE_COLOR_AGGRESSIVE, "Aggressive   (G_T -> 1.0)")
        y = self._blit_swatch_line(x, y, COLOR_FOOD_FRESH, "Food: fresh")
        y = self._blit_swatch_line(x, y, COLOR_FOOD_AGING, "Food: aging")
        y = self._blit_swatch_line(x, y, COLOR_FOOD_ROTTING, "Food: rotting (about to vanish)")
        y = self._blit_swatch_line(x, y, COLOR_GORILLA_ICON, "Gorilla troop (chimps blocked)")
        y = self._blit_marker_legend_line(x, y, "combat", "Combat / Attack event")
        y = self._blit_marker_legend_line(x, y, "birth", "Reproduction / birth event")
        y += 10

        # --- Controls ---
        y = self._blit_line(x, y, "CONTROLS", self.font_section, COLOR_HUD_ACCENT)
        y = self._blit_line(x, y, "SPACE       Pause / resume", self.font_small, COLOR_HUD_TEXT)
        y = self._blit_line(x, y, "UP / DOWN   Increase / decrease speed", self.font_small, COLOR_HUD_TEXT)
        y = self._blit_line(x, y, "S           Save checkpoint now", self.font_small, COLOR_HUD_TEXT)
        y = self._blit_line(x, y, "ESC         Quit", self.font_small, COLOR_HUD_TEXT)
        y += 10

        # --- Live status line(s) ---
        speed_text = f"Speed: {fps} steps/sec"
        y = self._blit_line(x, y, speed_text, self.font_stat, COLOR_HUD_TEXT)

        if paused:
            y = self._blit_line(x, y, "PAUSED", self.font_section, COLOR_HUD_PAUSED)

        if save_message:
            y = self._blit_line(x, y, save_message, self.font_stat, COLOR_HUD_SAVED)

    def _blit_line(
        self, x: int, y: int, text: str, font: pygame.font.Font, color: Tuple[int, int, int]
    ) -> int:
        surface = font.render(text, True, color)
        self.screen.blit(surface, (x, y))
        return y + surface.get_height() + 4

    def _blit_stat(self, x: int, y: int, label: str, value: str) -> int:
        line = f"  {label:<24} {value}"
        return self._blit_line(x, y, line, self.font_stat, COLOR_HUD_TEXT)

    def _blit_swatch_line(self, x: int, y: int, color: Tuple[int, int, int], label: str) -> int:
        swatch_rect = pygame.Rect(x + 2, y + 3, 14, 14)
        pygame.draw.rect(self.screen, color, swatch_rect, border_radius=3)
        pygame.draw.rect(self.screen, AGENT_OUTLINE_COLOR, swatch_rect, width=1, border_radius=3)
        text_surface = self.font_small.render(label, True, COLOR_HUD_TEXT)
        self.screen.blit(text_surface, (x + 24, y + 2))
        return y + max(swatch_rect.height, text_surface.get_height()) + 6

    def _blit_marker_legend_line(self, x: int, y: int, kind: str, label: str) -> int:
        marker_size = 18
        marker_surface = pygame.Surface((marker_size, marker_size), pygame.SRCALPHA)
        if kind == "combat":
            color = (*COLOR_COMBAT_FLASH, 255)
            pygame.draw.line(marker_surface, color, (2, 2), (marker_size - 2, marker_size - 2), 2)
            pygame.draw.line(marker_surface, color, (marker_size - 2, 2), (2, marker_size - 2), 2)
        else:
            color = (*COLOR_BIRTH_FLASH, 255)
            r = marker_size // 5
            cx, cy = marker_size // 2, marker_size // 2
            pygame.draw.circle(marker_surface, color, (cx - r // 2, cy - r // 2), r)
            pygame.draw.circle(marker_surface, color, (cx + r // 2, cy - r // 2), r)
            pygame.draw.polygon(
                marker_surface, color, [(cx - r, cy), (cx + r, cy), (cx, cy + r)]
            )
        self.screen.blit(marker_surface, (x + 2, y + 1))
        text_surface = self.font_small.render(label, True, COLOR_HUD_TEXT)
        self.screen.blit(text_surface, (x + 24, y + 2))
        return y + max(marker_size, text_surface.get_height()) + 6

    # ------------------------------------------------------------------
    # Full frame
    # ------------------------------------------------------------------
    def render(
        self,
        arena: CongoArena,
        flash_manager: FlashManager,
        fps: int,
        paused: bool,
        save_message: str,
    ) -> None:
        # Defensive check: this renderer assumes the ecosystem is exactly
        # GRID_COLS x GRID_ROWS. If that ever changes (e.g. a checkpoint
        # is loaded with a different grid size), fail loudly here rather
        # than silently drawing a subtly-misaligned scene.
        if arena.ecosystem.width != GRID_COLS or arena.ecosystem.height != GRID_ROWS:
            raise ValueError(
                f"SimulationRenderer is configured for a {GRID_COLS}x{GRID_ROWS} grid, "
                f"but the arena's ecosystem is {arena.ecosystem.width}x{arena.ecosystem.height}. "
                f"Update GRID_COLS/GRID_ROWS (and CELL_SIZE if needed) to match."
            )

        self.screen.fill(COLOR_BG)
        self._draw_terrain(arena)
        self._draw_gorillas(arena)
        self._draw_food(arena)
        self._draw_gorilla_cores(arena)
        self._draw_agents(arena)
        self._draw_flashes(arena, flash_manager)
        self._draw_hud(arena, fps=fps, paused=paused, save_message=save_message)
        pygame.display.flip()


# ----------------------------------------------------------------------
# Main application loop
# ----------------------------------------------------------------------
def _set_windows_dpi_awareness() -> None:
    """On Windows with display scaling enabled (125%/150%/200% — extremely
    common on modern 1080p+ laptops), an SDL/Pygame window that isn't
    marked DPI-aware gets upscaled by the OS compositor: a window we
    request at 1280x720 *logical* pixels can end up occupying a larger
    *physical* area than that, which is very likely the actual root
    cause of a window whose top gets cut off despite the requested
    height already looking reasonable. Marking the process DPI-aware
    (before any window is created) makes Windows report and honor true
    physical pixels instead, so our chosen WINDOW_WIDTH/HEIGHT means
    what it says. This is a no-op on any non-Windows platform.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        try:
            # PROCESS_SYSTEM_DPI_AWARE — preferred on Windows 8.1+.
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            # Fallback for older Windows versions lacking shcore.
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        # Never let a DPI-awareness quirk crash the whole visualization;
        # worst case, the window falls back to OS-scaled behavior.
        pass


def run_visualization(rng_seed: int = 2026, checkpoint_path: str = DEFAULT_CHECKPOINT_PATH) -> None:
    _set_windows_dpi_awareness()

    pygame.init()
    pygame.display.set_caption("Congo River Paradigm — Phase 5: Live Simulation")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    arena = build_arena(rng_seed=rng_seed)
    arena.reset(seed=rng_seed)

    renderer = SimulationRenderer(screen)
    flash_manager = FlashManager()

    fps = DEFAULT_FPS
    paused = False
    running = True

    save_message = ""
    save_message_expiry_ms = 0

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_UP:
                        fps = min(MAX_FPS, fps + FPS_STEP)
                    elif event.key == pygame.K_DOWN:
                        fps = max(MIN_FPS, fps - FPS_STEP)
                    elif event.key == pygame.K_s:
                        CheckpointManager.save(arena, checkpoint_path)
                        save_message = f"Checkpoint saved (step {arena.current_step})"
                        save_message_expiry_ms = pygame.time.get_ticks() + 2500

            if not running:
                break

            if not paused:
                positions_before_step = {
                    agent_id: (agent.x, agent.y)
                    for agent_id, agent in arena.agents_by_id.items()
                    if agent.is_alive
                }
                arena.step()
                flash_manager.process_step_log(arena.last_step_log, positions_before_step, arena)

            visible_save_message = (
                save_message if pygame.time.get_ticks() < save_message_expiry_ms else ""
            )

            renderer.render(
                arena,
                flash_manager,
                fps=fps,
                paused=paused,
                save_message=visible_save_message,
            )

            clock.tick(fps)
    except KeyboardInterrupt:
        # Allow Ctrl+C in the terminal to exit gracefully as well.
        pass
    finally:
        pygame.quit()
        sys.exit(0)


if __name__ == "__main__":
    run_visualization()