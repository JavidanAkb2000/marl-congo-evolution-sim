"""
src/environment/ecosystem.py

Core simulation logic for the Congo River Paradigm ecosystem.

This module implements the pure state/math layer of a 50x50 grid-based
ecosystem split into two territories by a virtual "river" barrier:

    - South Bank (Y <= 25): Bonobo territory. Resource abundant, food
      spawns frequently in clustered "fruit tree" patches with high
      energy value.
    - North Bank (Y > 25): Chimpanzee territory. Resource scarce, food
      spawns rarely, scattered randomly across the territory, with low
      energy value.

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
        north_spawn_prob: Per-step probability of a scattered scarcity
                           spawn event on the North Bank.
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
    """

    def __init__(
        self,
        width: int = 50,
        height: int = 50,
        river_y: int = 25,
        north_spawn_prob: float = 0.05,
        south_spawn_prob: float = 0.50,
        north_food_energy: float = 10.0,
        south_food_energy: float = 40.0,
        south_cluster_size_range: Tuple[int, int] = (3, 6),
        south_cluster_radius: int = 2,
        rng_seed: Optional[int] = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Grid width and height must be positive integers.")
        if not (0 <= river_y < height):
            raise ValueError("river_y must lie within the grid's vertical bounds.")

        self.width = width
        self.height = height
        self.river_y = river_y

        self.north_spawn_prob = north_spawn_prob
        self.south_spawn_prob = south_spawn_prob
        self.north_food_energy = north_food_energy
        self.south_food_energy = south_food_energy
        self.south_cluster_size_range = south_cluster_size_range
        self.south_cluster_radius = south_cluster_radius

        self._rng = random.Random(rng_seed)

        # Primary food store: a flat list of FoodItem objects.
        # Multi-occupancy is supported implicitly since several items
        # can share identical (x, y) coordinates.
        self.food_items: List[FoodItem] = []

        # Simple step counter, useful for logging/debugging and for
        # downstream RL loops that may want to track episode time.
        self.current_step: int = 0

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
        """Scarcity spawn logic for the North Bank (Chimpanzee territory).

        With probability `north_spawn_prob`, a single food item spawns
        at a fully random, scattered coordinate somewhere on the North
        Bank (y > river_y). Low resource density, low energy value.
        """
        if self._rng.random() >= self.north_spawn_prob:
            return

        x = self._rng.randint(0, self.width - 1)
        y = self._rng.randint(self.river_y + 1, self.height - 1)
        self.add_food(x=x, y=y, food_type="north", energy=self.north_food_energy)

    def _spawn_south_food(self) -> None:
        """Abundance spawn logic for the South Bank (Bonobo territory).

        With probability `south_spawn_prob`, a clustered "fruit tree"
        patch spawns on the South Bank (y <= river_y): a random cluster
        center is chosen, and several food items are scattered around
        it within `south_cluster_radius`, simulating fruit falling near
        a tree. High resource density, high energy value.
        """
        if self._rng.random() >= self.south_spawn_prob:
            return

        center_x = self._rng.randint(0, self.width - 1)
        center_y = self._rng.randint(0, self.river_y)

        min_items, max_items = self.south_cluster_size_range
        cluster_count = self._rng.randint(min_items, max_items)

        for _ in range(cluster_count):
            offset_x = self._rng.randint(-self.south_cluster_radius, self.south_cluster_radius)
            offset_y = self._rng.randint(-self.south_cluster_radius, self.south_cluster_radius)

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