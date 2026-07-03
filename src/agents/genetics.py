"""
src/agents/genetics.py

Phase 4, System 2: Genetic Algorithm & Reproduction Loop.

Implements the evolutionary machinery that runs at the end of every
environment step: fertility checks, mate matching, gene crossover,
Gaussian mutation, and metabolic reproduction costs.

Design note on gene mutation vs the Phase 3 invariant:
    Phase 3 established the hard constraint G_E + G_T == 1.0 for every
    agent. The spec for this phase asks for G_T and G_E to be mutated
    *independently* with Gaussian noise. To honor both requirements,
    this engine mutates each gene independently (as specified) and then
    re-normalizes the pair so they still sum to exactly 1.0, preserving
    the original invariant while still respecting independent mutation
    pressure on each trait.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

from src.agents.agent import ActionType, EvolvableAgent


class GeneticEngine:
    """Encapsulates fertility rules, mate selection, crossover and mutation.

    Attributes:
        reproduction_energy_threshold: Minimum energy for "Fertile" status.
        reproduction_cost: Energy deducted from each parent after mating.
        crossover_weight_parent1: Blend weight (0-1) given to parent 1's
                                   genes during crossover; parent 2 gets
                                   (1 - crossover_weight_parent1).
        mutation_rate: Probability [0, 1] that a given gene mutates.
        mutation_sigma: Standard deviation of the Gaussian mutation noise.
        offspring_initial_energy: Starting energy granted to newborns.
        mating_max_distance: Maximum Chebyshev distance between two
                              agents for them to be considered "adjacent"
                              and eligible to mate.
        north_hotspot_fertility_threshold: A LOWER fertility energy
                              threshold applied when an agent is
                              currently standing within a North Bank food
                              hotspot (see CongoEcosystem.north_hotspots),
                              letting Chimpanzees breed faster while
                              holding a productive patch to counteract
                              their generally higher mortality.
        north_hotspot_reproduction_cost: A LOWER reproduction energy
                              cost applied under the same hotspot
                              condition, for the same reason.
        low_population_threshold: When the TOTAL living population (as
                              reported by the caller) is at or below
                              this count, mating checks use
                              `low_population_mating_distance` instead
                              of `mating_max_distance`. This directly
                              targets the extinction-bottleneck/Allee
                              effect: once a population crashes to a
                              handful of individuals scattered across a
                              large grid, they may simply never wander
                              close enough to find each other under the
                              normal short-range distance, freezing the
                              population indefinitely (or, with aging
                              enabled, guaranteeing extinction). Giving
                              survivors a much longer "social attraction
                              range" when desperately few remain models
                              real crisis-response mate-seeking behavior.
        low_population_mating_distance: The expanded Chebyshev mating
                              distance used once the population is at
                              or below `low_population_threshold`.
    """

    def __init__(
        self,
        reproduction_energy_threshold: float = 80.0,
        reproduction_cost: float = 30.0,
        crossover_weight_parent1: float = 0.6,
        mutation_rate: float = 0.05,
        mutation_sigma: float = 0.05,
        offspring_initial_energy: float = 50.0,
        mating_max_distance: int = 1,
        north_hotspot_fertility_threshold: float = 50.0,
        north_hotspot_reproduction_cost: float = 12.0,
        low_population_threshold: int = 5,
        low_population_mating_distance: int = 15,
        rng: Optional[random.Random] = None,
    ) -> None:
        if not (0.0 <= crossover_weight_parent1 <= 1.0):
            raise ValueError("crossover_weight_parent1 must lie within [0.0, 1.0].")
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError("mutation_rate must lie within [0.0, 1.0].")

        self.reproduction_energy_threshold = reproduction_energy_threshold
        self.reproduction_cost = reproduction_cost
        self.crossover_weight_parent1 = crossover_weight_parent1
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.offspring_initial_energy = offspring_initial_energy
        self.mating_max_distance = mating_max_distance
        self.north_hotspot_fertility_threshold = north_hotspot_fertility_threshold
        self.north_hotspot_reproduction_cost = north_hotspot_reproduction_cost
        self.low_population_threshold = low_population_threshold
        self.low_population_mating_distance = low_population_mating_distance

        self._rng = rng if rng is not None else random.Random()

    # ------------------------------------------------------------------
    # Eligibility checks
    # ------------------------------------------------------------------
    def is_fertile(self, agent: EvolvableAgent, fertility_threshold: Optional[float] = None) -> bool:
        """An agent is Fertile if alive and energy exceeds the threshold.

        `fertility_threshold` lets a caller (e.g. the arena, for a North
        agent standing in a food hotspot) substitute a lower bar than
        the global default without mutating shared engine state.
        """
        threshold = (
            fertility_threshold if fertility_threshold is not None else self.reproduction_energy_threshold
        )
        return agent.is_alive and agent.energy > threshold

    def can_mate(
        self,
        agent1: EvolvableAgent,
        agent2: EvolvableAgent,
        action1: Optional[ActionType],
        action2: Optional[ActionType],
        threshold1: Optional[float] = None,
        threshold2: Optional[float] = None,
        current_population: Optional[int] = None,
    ) -> bool:
        """Determine whether two agents may reproduce this step.

        Requires: both alive, both Fertile (each judged against its own
        `threshold1`/`threshold2` override if supplied, else the global
        default), occupying the same or an adjacent cell (Chebyshev
        distance <= the applicable max distance), and both having
        selected a non-aggressive action (i.e. neither chose ATTACK).

        `current_population`, if supplied, lets the mating distance
        itself widen automatically once the total living population is
        at or below `low_population_threshold` — see the class
        docstring for the extinction-bottleneck rationale.
        """
        if agent1.agent_id == agent2.agent_id:
            return False
        if not (self.is_fertile(agent1, threshold1) and self.is_fertile(agent2, threshold2)):
            return False
        if action1 == ActionType.ATTACK or action2 == ActionType.ATTACK:
            return False

        max_distance = self.mating_max_distance
        if current_population is not None and current_population <= self.low_population_threshold:
            max_distance = self.low_population_mating_distance

        distance = max(abs(agent1.x - agent2.x), abs(agent1.y - agent2.y))
        return distance <= max_distance

    # ------------------------------------------------------------------
    # Crossover / mutation
    # ------------------------------------------------------------------
    @staticmethod
    def _blend(value1: float, value2: float, weight1: float) -> float:
        """Weighted crossover blend: value1 * w + value2 * (1 - w)."""
        return value1 * weight1 + value2 * (1.0 - weight1)

    def _mutate_gene(self, gene_value: float) -> float:
        """Apply Gaussian mutation with probability `mutation_rate`.

        On trigger, adds noise drawn from N(0, mutation_sigma) — i.e.
        a "+/- 0.05" style perturbation — and clips the result strictly
        to [0.0, 1.0].
        """
        if self._rng.random() < self.mutation_rate:
            gene_value += self._rng.gauss(0.0, self.mutation_sigma)
        return max(0.0, min(1.0, gene_value))

    def _crossover_and_mutate_genes(
        self, parent1: EvolvableAgent, parent2: EvolvableAgent
    ) -> Tuple[float, float]:
        """Produce a normalized (g_t, g_e) pair for the offspring.

        Both traits are crossed over via a weighted blend of the
        parents' genes, then independently mutated, then re-normalized
        so that g_t + g_e == 1.0 (preserving the Phase 3 invariant).
        """
        raw_g_t = self._blend(parent1.g_t, parent2.g_t, self.crossover_weight_parent1)
        raw_g_e = self._blend(parent1.g_e, parent2.g_e, self.crossover_weight_parent1)

        mutated_g_t = self._mutate_gene(raw_g_t)
        mutated_g_e = self._mutate_gene(raw_g_e)

        gene_sum = mutated_g_t + mutated_g_e
        if gene_sum <= 0.0:
            # Degenerate edge case (both genes mutated down to ~0):
            # fall back to a neutral 50/50 split.
            return 0.5, 0.5

        final_g_t = mutated_g_t / gene_sum
        final_g_e = mutated_g_e / gene_sum
        return final_g_t, final_g_e

    # ------------------------------------------------------------------
    # Reproduction
    # ------------------------------------------------------------------
    def reproduce(
        self,
        parent1: EvolvableAgent,
        parent2: EvolvableAgent,
        offspring_id: int,
        position: Tuple[int, int],
    ) -> EvolvableAgent:
        """Create a new offspring agent from two parents.

        Note: this method only constructs and returns the child; it
        does NOT deduct the reproduction cost from the parents. Call
        `apply_reproduction_cost()` on each parent separately (the
        caller/environment orchestrates the full transaction so it can
        log/inspect each step independently).
        """
        final_g_t, final_g_e = self._crossover_and_mutate_genes(parent1, parent2)
        offspring_generation = max(parent1.generation, parent2.generation) + 1

        child = EvolvableAgent(
            agent_id=offspring_id,
            x=position[0],
            y=position[1],
            forced_g_t=final_g_t,
            initial_energy=self.offspring_initial_energy,
            initial_stress=0.0,
            generation=offspring_generation,
            rng=self._rng,
        )
        # forced_g_t already derives g_e = 1 - g_t internally, which by
        # construction equals final_g_e since final_g_t + final_g_e == 1.
        return child

    def apply_reproduction_cost(self, parent: EvolvableAgent, cost: Optional[float] = None) -> None:
        """Deduct the metabolic cost of reproduction from a parent.

        `cost` lets a caller substitute a lower, hotspot-discounted cost
        without mutating shared engine state; defaults to `reproduction_cost`.
        """
        if not parent.is_alive:
            return
        actual_cost = cost if cost is not None else self.reproduction_cost
        parent.energy = max(0.0, min(100.0, parent.energy - actual_cost))
        parent.hunger = 100.0 - parent.energy
        if parent.energy <= 0.0:
            parent.is_alive = False
