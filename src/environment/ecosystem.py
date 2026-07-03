"""
src/environment/ecosystem.py

Core simulation logic for the Congo River Paradigm ecosystem.

This module implements the pure state/math layer of a 50x50 grid-based
ecosystem split into two territories by a virtual "river" barrier:

    - South Bank (Y <= 25): Bonobo territory. Resource abundant, food
      spawns frequently in clustered "fruit tree" patches with high
      energy value.
    - North Bank (Y > 25): Chimpanzee territory. Resource scarce and
      PATCHY — food spawns only within a handful of fixed geographic
      hotspots (never scattered evenly), forcing competing agents into
      the same choke-points, with low energy value per item.

No Pygame or any rendering dependency is imported here. This module is
render-agnostic and can be driven headlessly (e.g. for RL training) or
visualized by a separate rendering layer such as scripts/test_environment.py.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Tuple, Literal, Optional

# Type alias for bank/food classification
BankType = Literal["north", "south"]


@dataclass
class FoodItem:
    """Represents a single unit of food placed on the grid.

    Multiple FoodItem instances may share the exact same (x, y)
    coordinate, since the ecosystem grid supports multi-occupancy.
    """

    x: int
    y: int
    food_type: BankType
    energy: float

    def as_tuple(self) -> Tuple[int, int, BankType, float]:
        """Return a plain tuple representation of this food item."""
        return (self.x, self.y, self.food_type, self.energy)


class CongoEcosystem:
    """Core state and stepping logic for the Congo River ecosystem.

    Attributes:
        width: Grid width in cells.
        height: Grid height in cells.
        river_y: The row index that divides North Bank from South Bank.
                 South Bank is defined as y <= river_y, North Bank as
                 y > river_y.
        north_spawn_prob: Per-step probability, evaluated INDEPENDENTLY
                           for EACH North hotspot, that it fires a
                           spawn event this step (so multiple hotspots
                           can fire in the same step). Combined with
                           `north_hotspot_spawn_size` and
                           `north_food_energy`, this determines the
                           North bank's total food throughput, which is
                           calibrated to actually be able to feed the
                           population living around the hotspots.
        south_spawn_prob: Per-step probability of a clustered abundance
                           spawn event on the South Bank.
        north_food_energy: Energy value granted by a single North Bank
                            food item.
        south_food_energy: Energy value granted by a single South Bank
                            food item.
        south_cluster_size_range: Inclusive (min, max) number of food
                                   items generated per South Bank spawn
                                   event (simulating a fruit tree patch).
        south_cluster_radius: Maximum Chebyshev distance from a cluster
                               center that a clustered food item can
                               spawn at.
        north_hotspot_count: Number of fixed, dense food "hotspots"
                              (e.g. isolated fruiting trees, termite
                              mounds) scattered across North territory
                              at construction time. ALL North Bank food
                              spawns only within `north_hotspot_radius`
                              of one of these points — the rest of the
                              North is a hard desert. This deliberately
                              forces competing Chimpanzees into a
                              handful of choke-points (real-world
                              "resource patchiness" ecology), driving
                              territorial combat while still guaranteeing
                              concentrated, reachable food for whichever
                              agents dominate a given patch.
        north_hotspot_radius: Chebyshev radius around a hotspot center
                               within which North food can spawn.
        north_hotspot_spawn_size: Inclusive (min, max) number of food
                                   items dropped at a hotspot per
                                   triggered North spawn event.
        south_isolated_fruit_chance: Probability that a triggered South
                                      Bank spawn event drops a single
                                      isolated scattered fruit (anywhere
                                      on the South bank) instead of a
                                      full fruit-tree cluster, breaking
                                      up the purely-clustered look with
                                      more naturally scattered filler.
    """

    def __init__(
        self,
        width: int = 50,
        height: int = 50,
        river_y: int = 25,
        north_spawn_prob: float = 0.25,
        south_spawn_prob: float = 0.50,
        north_food_energy: float = 35.0,
        south_food_energy: float = 40.0,
        south_cluster_size_range: Tuple[int, int] = (2, 4),
        south_cluster_radius: int = 2,
        north_hotspot_count: int = 4,
        north_hotspot_radius: int = 3,
        north_hotspot_spawn_size: Tuple[int, int] = (3, 6),
        south_isolated_fruit_chance: float = 0.30,
        rng_seed: Optional[int] = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Grid width and height must be positive integers.")
        if not (0 <= river_y < height):
            raise ValueError("river_y must lie within the grid's vertical bounds.")
        if north_hotspot_count < 1:
            raise ValueError("north_hotspot_count must be at least 1.")

        self.width = width
        self.height = height
        self.river_y = river_y

        self.north_spawn_prob = north_spawn_prob
        self.south_spawn_prob = south_spawn_prob
        self.north_food_energy = north_food_energy
        self.south_food_energy = south_food_energy
        self.south_cluster_size_range = south_cluster_size_range
        self.south_cluster_radius = south_cluster_radius
        self.north_hotspot_count = north_hotspot_count
        self.north_hotspot_radius = north_hotspot_radius
        self.north_hotspot_spawn_size = north_hotspot_spawn_size
        self.south_isolated_fruit_chance = south_isolated_fruit_chance

        self._rng = random.Random(rng_seed)

        # Fixed North Bank food-hotspot geography. Generated ONCE here
        # (not regenerated by reset()) so a given ecosystem instance has
        # stable "terrain" across resets, the same way real geography
        # (a particular fruiting tree's location) doesn't move — only
        # the food items themselves come and go. Can be overwritten
        # directly (e.g. by CheckpointManager.load) to restore an exact
        # previously-generated layout.
        self.north_hotspots: List[Tuple[int, int]] = self._generate_north_hotspots()

        # Primary food store: a flat list of FoodItem objects.
        # Multi-occupancy is supported implicitly since several items
        # can share identical (x, y) coordinates.
        self.food_items: List[FoodItem] = []

        # Simple step counter, useful for logging/debugging and for
        # downstream RL loops that may want to track episode time.
        self.current_step: int = 0

    def _generate_north_hotspots(self) -> List[Tuple[int, int]]:
        """Randomly place `north_hotspot_count` fixed food hotspots in North territory."""
        hotspots = []
        for _ in range(self.north_hotspot_count):
            x = self._rng.randint(0, self.width - 1)
            y = self._rng.randint(self.river_y + 1, self.height - 1)
            hotspots.append((x, y))
        return hotspots

    def is_in_north_hotspot(self, x: int, y: int) -> bool:
        """True if (x, y) lies within `north_hotspot_radius` of any hotspot."""
        for hotspot_x, hotspot_y in self.north_hotspots:
            if max(abs(x - hotspot_x), abs(y - hotspot_y)) <= self.north_hotspot_radius:
                return True
        return False

    # ------------------------------------------------------------------
    # Boundary / territory helpers
    # ------------------------------------------------------------------
    def is_within_bounds(self, x: int, y: int) -> bool:
        """Return True if (x, y) lies within the hard grid boundaries."""
        return 0 <= x < self.width and 0 <= y < self.height

    def get_bank(self, y: int) -> BankType:
        """Classify a row index as belonging to the North or South Bank.

        South Bank: y <= river_y
        North Bank: y > river_y
        """
        if not (0 <= y < self.height):
            raise ValueError(f"y={y} is outside the grid's vertical range [0, {self.height}).")
        return "south" if y <= self.river_y else "north"

    def is_move_legal(self, current_y: int, target_y: int) -> bool:
        """Validate that a move does not cross the Congo River barrier.

        An entity may move freely within its own bank (including along
        the river's edge) but can never cross from North to South or
        vice versa. Movement that would leave the hard grid boundary
        entirely is also considered illegal.

        Args:
            current_y: The entity's current row coordinate.
            target_y: The row coordinate the entity wishes to move to.

        Returns:
            True if the move keeps the entity within grid bounds and on
            the same bank; False otherwise.
        """
        if not (0 <= target_y < self.height):
            return False
        if not (0 <= current_y < self.height):
            return False
        return self.get_bank(current_y) == self.get_bank(target_y)

    def clamp_to_bounds(self, x: int, y: int) -> Tuple[int, int]:
        """Clamp a coordinate to the hard grid boundaries."""
        clamped_x = max(0, min(self.width - 1, x))
        clamped_y = max(0, min(self.height - 1, y))
        return clamped_x, clamped_y

    # ------------------------------------------------------------------
    # Food / state management
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all food and reset the step counter to a fresh state."""
        self.food_items.clear()
        self.current_step = 0

    def add_food(self, x: int, y: int, food_type: BankType, energy: float) -> FoodItem:
        """Add a single food item to the grid at (x, y).

        Multiple food items may coexist at the same coordinate; this
        method does not check for or prevent overlap.

        Raises:
            ValueError: If (x, y) is outside the hard grid boundaries.
        """
        if not self.is_within_bounds(x, y):
            raise ValueError(f"Cannot place food at out-of-bounds coordinate ({x}, {y}).")

        item = FoodItem(x=x, y=y, food_type=food_type, energy=energy)
        self.food_items.append(item)
        return item

    def get_food_at(self, x: int, y: int) -> List[FoodItem]:
        """Return all food items currently occupying coordinate (x, y)."""
        return [item for item in self.food_items if item.x == x and item.y == y]

    def consume_food_at(self, x: int, y: int, max_items: Optional[int] = None) -> float:
        """Consume food located at (x, y), removing it from the grid.

        Args:
            x: Target column coordinate.
            y: Target row coordinate.
            max_items: Optional cap on how many food items to consume in
                       this call. If None, all food at the coordinate is
                       consumed at once.

        Returns:
            The total energy value gained from the consumed food items.
            Returns 0.0 if no food was present at the coordinate.
        """
        matches = [item for item in self.food_items if item.x == x and item.y == y]
        if not matches:
            return 0.0

        if max_items is not None:
            matches = matches[:max_items]

        total_energy = sum(item.energy for item in matches)
        matched_ids = {id(item) for item in matches}
        self.food_items = [item for item in self.food_items if id(item) not in matched_ids]
        return total_energy

    def get_food_matrix(self) -> List[List[int]]:
        """Return a (height x width) matrix of food item counts per cell.

        matrix[y][x] gives the number of food items currently occupying
        grid coordinate (x, y). This is convenient for both RL
        observation encoding and quick debugging/rendering.
        """
        matrix = [[0 for _ in range(self.width)] for _ in range(self.height)]
        for item in self.food_items:
            matrix[item.y][item.x] += 1
        return matrix

    def get_energy_matrix(self) -> List[List[float]]:
        """Return a (height x width) matrix of total energy per cell."""
        matrix = [[0.0 for _ in range(self.width)] for _ in range(self.height)]
        for item in self.food_items:
            matrix[item.y][item.x] += item.energy
        return matrix

    def get_food_by_type(self, food_type: BankType) -> List[FoodItem]:
        """Return all current food items of a given type ('north'/'south')."""
        return [item for item in self.food_items if item.food_type == food_type]

    # ------------------------------------------------------------------
    # Resource spawning engine
    # ------------------------------------------------------------------
    def _spawn_north_food(self) -> None:
        """Patchy, choke-point scarcity spawn logic for the North Bank.

        CALIBRATION NOTE (carrying-capacity rebalance): each of the
        `north_hotspots` now rolls its `north_spawn_prob` INDEPENDENTLY
        every step (this used to be a single global roll that picked
        ONE hotspot at most per step). With the default 4 hotspots at
        `north_spawn_prob=0.25`, expected firing hotspots per step =
        4 x 0.25 = 1.0; each firing hotspot drops an average of
        (3+6)/2 = 4.5 items at the boosted `north_food_energy` (35.0),
        for an expected combined output of ~157.5 energy/step across
        the whole North bank — calibrated against a ~75-90 agent North
        population's baseline metabolic demand (see build_arena()).
        Everywhere outside a hotspot radius remains a hard desert; this
        is still a deliberate "resource patchiness" design that forces
        competing agents into a handful of choke-points, but the
        *combined* throughput of those choke-points is now sized to
        actually be able to feed the population living around them,
        rather than mathematically guaranteeing mass starvation
        regardless of behavior (which was the actual root cause behind
        the earlier fast North extinctions — confirmed via a controlled
        test where even a 100%-cooperative, zero-aggression North
        population still collapsed from pure resource math).
        """
        for hotspot_x, hotspot_y in self.north_hotspots:
            if self._rng.random() >= self.north_spawn_prob:
                continue

            min_items, max_items = self.north_hotspot_spawn_size
            item_count = self._rng.randint(min_items, max_items)

            for _ in range(item_count):
                offset_x = self._rng.randint(-self.north_hotspot_radius, self.north_hotspot_radius)
                offset_y = self._rng.randint(-self.north_hotspot_radius, self.north_hotspot_radius)

                raw_x = hotspot_x + offset_x
                raw_y = hotspot_y + offset_y

                clamped_x, clamped_y = self.clamp_to_bounds(raw_x, raw_y)
                # Never let a hotspot's offset spill south across the river.
                clamped_y = max(clamped_y, self.river_y + 1)

                self.add_food(x=clamped_x, y=clamped_y, food_type="north", energy=self.north_food_energy)

    def _spawn_south_food(self) -> None:
        """Abundance spawn logic for the South Bank (Bonobo territory).

        With probability `south_spawn_prob`, a South Bank spawn event
        triggers. Each triggered event is one of two kinds:

          1. Isolated fruit (probability `south_isolated_fruit_chance`):
             a single food item scattered anywhere on the South bank,
             independent of any cluster center. This fills in the gaps
             between fruit-tree clusters with more naturally distributed
             "background" food instead of leaving hard dead zones.

          2. A fruit-tree cluster (otherwise): a random cluster center
             is chosen, and several food items scatter around it using
             a *Gaussian* offset (not a uniform square block), which
             produces a soft, organically-tapering clump — dense near
             the center, naturally thinning at the edges — rather than
             the blocky, grid-aligned "geometric" look a uniform square
             offset produces at small radii.
        """
        if self._rng.random() >= self.south_spawn_prob:
            return

        if self._rng.random() < self.south_isolated_fruit_chance:
            x = self._rng.randint(0, self.width - 1)
            y = self._rng.randint(0, self.river_y)
            self.add_food(x=x, y=y, food_type="south", energy=self.south_food_energy)
            return

        center_x = self._rng.randint(0, self.width - 1)
        center_y = self._rng.randint(0, self.river_y)

        min_items, max_items = self.south_cluster_size_range
        cluster_count = self._rng.randint(min_items, max_items)

        # A Gaussian standard deviation derived from the configured
        # radius gives a natural, rounded falloff instead of a uniform
        # square block of discrete offsets.
        sigma = max(1.0, self.south_cluster_radius / 1.4)
        max_spread = self.south_cluster_radius * 3  # soft cap on rare Gaussian tail outliers

        for _ in range(cluster_count):
            offset_x = int(round(self._rng.gauss(0.0, sigma)))
            offset_y = int(round(self._rng.gauss(0.0, sigma)))
            offset_x = max(-max_spread, min(max_spread, offset_x))
            offset_y = max(-max_spread, min(max_spread, offset_y))

            raw_x = center_x + offset_x
            raw_y = center_y + offset_y

            # Keep clustered spawns strictly within bounds AND strictly
            # within the South Bank (never let a cluster spill across
            # the river due to the offset).
            clamped_x, clamped_y = self.clamp_to_bounds(raw_x, raw_y)
            clamped_y = min(clamped_y, self.river_y)

            self.add_food(
                x=clamped_x,
                y=clamped_y,
                food_type="south",
                energy=self.south_food_energy,
            )

    def step(self) -> None:
        """Advance the ecosystem by one simulation tick.

        Runs the resource spawning engine for both banks independently
        (scarcity scatter spawn on the North, abundance cluster spawn on
        the South) and increments the internal step counter.
        """
        self._spawn_north_food()
        self._spawn_south_food()
        self.current_step += 1

    # ------------------------------------------------------------------
    # Convenience / introspection
    # ------------------------------------------------------------------
    def food_count(self) -> int:
        """Return the total number of food items currently on the grid."""
        return len(self.food_items)

    def __repr__(self) -> str:
        return (
            f"CongoEcosystem(width={self.width}, height={self.height}, "
            f"river_y={self.river_y}, step={self.current_step}, "
            f"food_count={self.food_count()})"
        )
