"""
src/persistence/checkpoint.py

Phase 4, System 3: Checkpoint System (State Persistence).

Serializes and restores the exact state of a CongoArena world: the
step counter, the full ecosystem food grid, the ID counter used to
guarantee unique future agent IDs, and every agent's complete internal
state (position, genes, energy/hunger/stress, generation, alive flag).

Format: structured JSON. JSON was chosen over pickle deliberately:
    - It's human-readable and diffable, which matters a great deal for
      debugging a long-running evolutionary simulation.
    - It's safe to load from untrusted or shared sources (unlike
      pickle, which can execute arbitrary code on load).
    - It's portable across Python versions/interpreters.
The trade-off is a small amount of explicit (de)serialization code
below, which is a good trade for a "checkpoint you can inspect and
trust" over a slightly-more-convenient pickle blob.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from src.agents.agent import EvolvableAgent
from src.agents.genetics import GeneticEngine
from src.environment.arena import CongoArena
from src.environment.ecosystem import CongoEcosystem
from src.environment.interactions import InteractionResolver

CHECKPOINT_FORMAT_VERSION = 1


class CheckpointManager:
    """Static utility class for saving/loading CongoArena world state."""

    @staticmethod
    def save(arena: CongoArena, filepath: str) -> None:
        """Serialize the arena's full world state to a JSON file.

        Args:
            arena: The CongoArena instance to snapshot.
            filepath: Destination path. Parent directories are created
                      automatically if they don't already exist.
        """
        ecosystem = arena.ecosystem

        data = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "step": arena.current_step,
            "next_agent_id": arena._next_agent_id,
            "ecosystem": {
                "width": ecosystem.width,
                "height": ecosystem.height,
                "river_y": ecosystem.river_y,
                "north_spawn_prob": ecosystem.north_spawn_prob,
                "south_spawn_prob": ecosystem.south_spawn_prob,
                "north_food_energy": ecosystem.north_food_energy,
                "south_food_energy": ecosystem.south_food_energy,
                "south_cluster_size_range": list(ecosystem.south_cluster_size_range),
                "south_cluster_radius": ecosystem.south_cluster_radius,
                "north_hotspot_count": ecosystem.north_hotspot_count,
                "north_hotspot_radius": ecosystem.north_hotspot_radius,
                "north_hotspot_spawn_size": list(ecosystem.north_hotspot_spawn_size),
                "north_hotspots": [list(coord) for coord in ecosystem.north_hotspots],
                "south_isolated_fruit_chance": ecosystem.south_isolated_fruit_chance,
                "gorilla_occupied_count": ecosystem.gorilla_occupied_count,
                "depletion_threshold": ecosystem.depletion_threshold,
                "depletion_recovery_steps": ecosystem.depletion_recovery_steps,
                "gorilla_forage_rate": ecosystem.gorilla_forage_rate,
                "gorilla_migration_threshold": ecosystem.gorilla_migration_threshold,
                "gorilla_min_residence_steps": ecosystem.gorilla_min_residence_steps,
                "midway_food_prob": ecosystem.midway_food_prob,
                "midway_food_energy": ecosystem.midway_food_energy,
                "midway_food_max_per_step": ecosystem.midway_food_max_per_step,
                "food_decay_steps": ecosystem.food_decay_steps,
                "hotspot_state": [dict(s) for s in ecosystem.hotspot_state],
                "food_items": [
                    {
                        "x": item.x,
                        "y": item.y,
                        "food_type": item.food_type,
                        "energy": item.energy,
                        "spawn_step": item.spawn_step,
                    }
                    for item in ecosystem.food_items
                ],
            },
            "agents": [
                {
                    "agent_id": agent.agent_id,
                    "x": agent.x,
                    "y": agent.y,
                    "g_e": agent.g_e,
                    "g_t": agent.g_t,
                    "g_size": agent.g_size,
                    "g_fertility": agent.g_fertility,
                    "reproduction_cooldown": agent.reproduction_cooldown,
                    "energy": agent.energy,
                    "hunger": agent.hunger,
                    "stress": agent.stress,
                    "generation": agent.generation,
                    "age": agent.age,
                    "max_age": agent.max_age,
                    "is_alive": agent.is_alive,
                }
                for agent in arena.agents_by_id.values()
            ],
            "possible_agents": list(arena.possible_agents),
        }

        directory = os.path.dirname(os.path.abspath(filepath))
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(
        filepath: str,
        genetic_engine: Optional[GeneticEngine] = None,
        interaction_resolver: Optional[InteractionResolver] = None,
        perception_radius: int = 3,
        max_steps: Optional[int] = None,
        rng_seed: Optional[int] = None,
    ) -> CongoArena:
        """Rebuild a fully-restored CongoArena from a checkpoint file.

        Reconstructs the CongoEcosystem (including every food item on
        the grid), re-instantiates every agent with its exact saved
        position, genes, bio-energetic state, and generation count, and
        restores the arena's step counter and unique-ID sequence so the
        simulation can resume seamlessly.

        Args:
            filepath: Path to a JSON checkpoint written by `save()`.
            genetic_engine: Optional custom GeneticEngine for the
                             restored arena (a default is constructed
                             if omitted).
            interaction_resolver: Optional custom InteractionResolver
                                    (a default is constructed if omitted).
            perception_radius: Perception radius for the restored arena.
            max_steps: Optional episode length cap for the restored arena.
            rng_seed: Optional RNG seed for the restored arena's shared
                      random.Random instance.

        Returns:
            A fully-populated CongoArena, ready to have `.step()` called
            on it immediately (no `.reset()` call is needed or wanted,
            since that would discard the restored state).
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint format_version={data.get('format_version')!r}; "
                f"expected {CHECKPOINT_FORMAT_VERSION}."
            )

        eco_data = data["ecosystem"]
        ecosystem = CongoEcosystem(
            width=eco_data["width"],
            height=eco_data["height"],
            river_y=eco_data["river_y"],
            north_spawn_prob=eco_data["north_spawn_prob"],
            south_spawn_prob=eco_data["south_spawn_prob"],
            north_food_energy=eco_data["north_food_energy"],
            south_food_energy=eco_data["south_food_energy"],
            south_cluster_size_range=tuple(eco_data["south_cluster_size_range"]),
            south_cluster_radius=eco_data["south_cluster_radius"],
            # .get(...) with a default preserves backward compatibility
            # with checkpoints saved before these tunables existed.
            north_hotspot_count=eco_data.get("north_hotspot_count", 4),
            north_hotspot_radius=eco_data.get("north_hotspot_radius", 3),
            north_hotspot_spawn_size=tuple(eco_data.get("north_hotspot_spawn_size", (1, 3))),
            south_isolated_fruit_chance=eco_data.get("south_isolated_fruit_chance", 0.30),
            gorilla_occupied_count=eco_data.get("gorilla_occupied_count", 2),
            depletion_threshold=eco_data.get("depletion_threshold", 400.0),
            depletion_recovery_steps=eco_data.get("depletion_recovery_steps", 40),
            gorilla_forage_rate=eco_data.get("gorilla_forage_rate", 3.0),
            gorilla_migration_threshold=eco_data.get("gorilla_migration_threshold", 600.0),
            gorilla_min_residence_steps=eco_data.get("gorilla_min_residence_steps", 150),
            midway_food_prob=eco_data.get("midway_food_prob", 0.15),
            midway_food_energy=eco_data.get("midway_food_energy", 4.0),
            midway_food_max_per_step=eco_data.get("midway_food_max_per_step", 2),
            food_decay_steps=eco_data.get("food_decay_steps", 100),
            rng_seed=rng_seed,
        )
        # Restore the ecosystem's step counter BEFORE re-adding food, so
        # any item without a recorded spawn_step (old checkpoint) is
        # stamped with the current step rather than 0 (which would make it
        # instantly rot on resume).
        ecosystem.current_step = data["step"]
        # The constructor already generated a fresh, randomized set of
        # hotspots. If this checkpoint recorded specific hotspot
        # coordinates (the normal case), overwrite them so the restored
        # ecosystem's food "geography" is IDENTICAL to the saved one,
        # not just statistically similar.
        if "north_hotspots" in eco_data:
            ecosystem.north_hotspots = [tuple(coord) for coord in eco_data["north_hotspots"]]
        # Restore per-hotspot dynamic state (gorilla flags, depletion,
        # recovery timers) so a resumed run continues the exact same patch
        # cycle rather than resetting every patch to fresh.
        if "hotspot_state" in eco_data:
            ecosystem.hotspot_state = [dict(s) for s in eco_data["hotspot_state"]]

        for item in eco_data["food_items"]:
            ecosystem.add_food(
                x=item["x"],
                y=item["y"],
                food_type=item["food_type"],
                energy=item["energy"],
                # Preserve exact remaining freshness; fall back to the
                # current step for pre-decay checkpoints (treated as fresh).
                spawn_step=item.get("spawn_step", ecosystem.current_step),
            )

        # initial_population=0: population is restored explicitly below,
        # not via reset() (which would wipe the ecosystem/food we just
        # rebuilt above).
        arena = CongoArena(
            ecosystem=ecosystem,
            initial_population=0,
            perception_radius=perception_radius,
            genetic_engine=genetic_engine,
            interaction_resolver=interaction_resolver,
            max_steps=max_steps,
            rng_seed=rng_seed,
        )
        arena.current_step = data["step"]
        arena._next_agent_id = data["next_agent_id"]
        arena.possible_agents = list(data.get("possible_agents", []))

        for agent_data in data["agents"]:
            agent = EvolvableAgent(
                agent_id=agent_data["agent_id"],
                x=agent_data["x"],
                y=agent_data["y"],
                forced_g_t=agent_data["g_t"],
                initial_energy=agent_data["energy"],
                initial_stress=agent_data["stress"],
                generation=agent_data["generation"],
                age=agent_data.get("age", 0),
                max_age=agent_data.get("max_age", None),
                g_size=agent_data.get("g_size", 1.0),
                g_fertility=agent_data.get("g_fertility", 0.5),
                reproduction_cooldown=agent_data.get("reproduction_cooldown", 0),
                rng=arena._rng,
            )
            # forced_g_t already reproduces g_e = 1 - g_t; explicitly
            # restore hunger/is_alive as well in case of any edge-case
            # divergence, so the restored state matches the saved state
            # byte-for-byte on every tracked field.
            agent.hunger = agent_data["hunger"]
            agent.is_alive = agent_data["is_alive"]

            arena.agents_by_id[agent.agent_id] = agent
            name = arena._agent_name(agent.agent_id)
            if name not in arena.possible_agents:
                arena.possible_agents.append(name)

        arena.agents = [
            arena._agent_name(a.agent_id) for a in arena.agents_by_id.values() if a.is_alive
        ]
        arena.rewards = {name: 0.0 for name in arena.agents}
        arena.terminations = {name: False for name in arena.agents}
        arena.truncations = {name: False for name in arena.agents}
        arena.infos = {name: {} for name in arena.agents}

        return arena