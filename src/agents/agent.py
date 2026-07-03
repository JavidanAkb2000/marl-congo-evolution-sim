"""
src/agents/agent.py

Phase 3: Digital DNA and Emotional Brain.

Defines EvolvableAgent, an individual actor in the Congo River Paradigm
simulation. Each agent carries a two-gene "digital chromosome"
(Eros / Thanatos), a bio-energetic metabolic loop (energy, hunger,
stress), and a lightweight, dependency-free Fuzzy Logic Controller that
acts as its "emotional brain" for action selection.

No external fuzzy-logic libraries are used. All membership functions
and rule aggregation are implemented in pure Python for maximum
portability and stability.
"""

from __future__ import annotations

import random
from enum import Enum
from typing import Dict, Optional, Tuple


class ActionType(Enum):
    """The discrete action spectrum available to an agent."""

    IDLE = "IDLE"
    MOVE = "MOVE"
    EAT = "EAT"
    ATTACK = "ATTACK"
    COOPERATE = "COOPERATE"


class EvolvableAgent:
    """A single evolvable organism in the Congo River ecosystem.

    Digital Chromosome:
        g_t (Thanatos): Innate aggression / war drive, in [0.0, 1.0].
        g_e (Eros): Innate cooperation / peace drive, in [0.0, 1.0].
        Constraint: g_e + g_t == 1.0 at all times (fixed at birth; this
        implementation does not currently support gene mutation, which
        is left for the evolutionary/reproduction phase).

    Bio-Energetic State:
        energy: [0, 100], starts at 100. Reaching 0 kills the agent.
        hunger: [0, 100], always derived as (100 - energy).
        stress: [0, 100], updated from local environmental context
                (rival proximity, starvation level) via update_stress().

    Emotional Brain:
        A pure-Python Mamdani-style fuzzy controller. Hunger and stress
        are fuzzified into Low/Medium/High membership degrees using
        trapezoidal membership functions. A hand-authored rule base
        combines these with the agent's genetic traits (and simple
        environmental context flags) to produce a crisp weighted score
        per action, which is then normalized into an action-preference
        distribution over [MOVE, EAT, ATTACK, COOPERATE].
    """

    # ------------------------------------------------------------------
    # Class-level constants
    # ------------------------------------------------------------------

    # Metabolic energy cost incurred by each action per simulation step.
    ENERGY_COST: Dict[ActionType, float] = {
        ActionType.IDLE: 1.0,
        ActionType.MOVE: 2.0,
        ActionType.ATTACK: 10.0,
        ActionType.EAT: 1.0,       # Minimal cost: the act of feeding itself.
        ActionType.COOPERATE: 1.0,  # Minimal cost: a social interaction.
    }

    # Trapezoidal membership function parameters (a, b, c, d) shared by
    # both Hunger and Stress, since both are normalized on a [0, 100]
    # scale with identical linguistic semantics (Low / Medium / High).
    #   Full membership (degree 1.0) on the flat [b, c] plateau,
    #   ramping linearly to 0 outside [a, d].
    MF_LOW: Tuple[float, float, float, float] = (-1.0, 0.0, 20.0, 45.0)
    MF_MEDIUM: Tuple[float, float, float, float] = (25.0, 45.0, 55.0, 75.0)
    MF_HIGH: Tuple[float, float, float, float] = (55.0, 80.0, 100.0, 101.0)

    # Small non-zero baseline so that every action retains a nonzero
    # probability even when no rule fires strongly (keeps the policy
    # exploratory rather than degenerate).
    BASELINE_WEIGHT: float = 0.05

    def __init__(
        self,
        agent_id: int,
        x: int,
        y: int,
        forced_g_t: Optional[float] = None,
        initial_energy: float = 100.0,
        initial_stress: float = 0.0,
        generation: int = 0,
        age: int = 0,
        max_age: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        """Initialize a new EvolvableAgent.

        Args:
            agent_id: Unique identifier for this agent.
            x: Initial grid column.
            y: Initial grid row.
            forced_g_t: Optional override for the Thanatos gene, useful
                        for constructing deterministic test/scenario
                        agents. If None, G_T is drawn stochastically
                        from Uniform(0.0, 1.0) at birth.
            initial_energy: Starting energy value (0-100). Defaults to
                             a full 100, but can be lowered to simulate
                             an already-starved test profile.
            initial_stress: Starting stress value (0-100).
            generation: Evolutionary generation counter. 0 for agents
                        spawned into the initial population; offspring
                        created by the genetic algorithm inherit
                        max(parent_generations) + 1.
            age: Number of simulation ticks this agent has already
                 lived through. Normally starts at 0; exposed as a
                 constructor argument so a checkpoint reload can restore
                 an agent's exact accumulated age.
            max_age: Lifespan cap in simulation ticks. Once `age` reaches
                      this value, `is_expired()` returns True and the
                      environment is expected to retire the agent (an
                      "old-age" death, independent of energy). None means
                      no cap — the agent can theoretically live forever,
                      which is also the default for full backward
                      compatibility with any code that doesn't set this.
            rng: Optional shared random.Random instance for reproducible
                 experiments. A private instance is created if omitted.
        """
        self._rng = rng if rng is not None else random.Random()

        self.agent_id = agent_id
        self.x = x
        self.y = y
        self.generation = generation
        self.age = age
        self.max_age = max_age

        # --- Digital Chromosome ---
        if forced_g_t is not None:
            if not (0.0 <= forced_g_t <= 1.0):
                raise ValueError("forced_g_t must lie within [0.0, 1.0].")
            self.g_t = float(forced_g_t)
        else:
            self.g_t = self._rng.uniform(0.0, 1.0)
        self.g_e = 1.0 - self.g_t

        # --- Bio-Energetic State ---
        self.energy = max(0.0, min(100.0, initial_energy))
        self.hunger = 100.0 - self.energy
        self.stress = max(0.0, min(100.0, initial_stress))

        self.is_alive = self.energy > 0.0

        # --- Introspection caches (populated after decide_action /
        # compute_action_weights, useful for logging & test harnesses) ---
        self.last_fuzzy_hunger: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.last_fuzzy_stress: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.last_action_weights: Dict[ActionType, float] = {}
        self.last_action_probabilities: Dict[ActionType, float] = {}
        self.last_chosen_action: Optional[ActionType] = None

    # ------------------------------------------------------------------
    # Lifespan / aging
    # ------------------------------------------------------------------
    def increment_age(self, amount: int = 1) -> None:
        """Advance this agent's age by one simulation tick (or `amount`)."""
        self.age += amount

    def is_expired(self) -> bool:
        """True once this agent has reached its lifespan cap (old-age death).

        Always False if `max_age` is None (no cap configured).
        """
        return self.max_age is not None and self.age >= self.max_age

    # ------------------------------------------------------------------
    # Fuzzy membership functions
    # ------------------------------------------------------------------
    @staticmethod
    def _trapezoid(x: float, params: Tuple[float, float, float, float]) -> float:
        """Evaluate a trapezoidal membership function at point x.

        params = (a, b, c, d) describes a trapezoid that ramps up
        linearly from 0 at `a` to 1 at `b`, stays at 1 until `c`, then
        ramps back down to 0 at `d`.
        """
        a, b, c, d = params
        if x <= a or x >= d:
            return 0.0
        if a < x < b:
            return (x - a) / (b - a)
        if b <= x <= c:
            return 1.0
        # c < x < d
        return (d - x) / (d - c)

    def fuzzify_hunger(self, hunger: Optional[float] = None) -> Tuple[float, float, float]:
        """Return (low, medium, high) membership degrees for hunger."""
        h = self.hunger if hunger is None else hunger
        low = self._trapezoid(h, self.MF_LOW)
        medium = self._trapezoid(h, self.MF_MEDIUM)
        high = self._trapezoid(h, self.MF_HIGH)
        return low, medium, high

    def fuzzify_stress(self, stress: Optional[float] = None) -> Tuple[float, float, float]:
        """Return (low, medium, high) membership degrees for stress."""
        s = self.stress if stress is None else stress
        low = self._trapezoid(s, self.MF_LOW)
        medium = self._trapezoid(s, self.MF_MEDIUM)
        high = self._trapezoid(s, self.MF_HIGH)
        return low, medium, high

    # ------------------------------------------------------------------
    # Fuzzy rule base / defuzzification
    # ------------------------------------------------------------------
    def _evaluate_rule_base(
        self,
        h_low: float,
        h_med: float,
        h_high: float,
        s_low: float,
        s_med: float,
        s_high: float,
        food_present: bool,
        rival_present: bool,
    ) -> Dict[ActionType, float]:
        """Fire the linguistic rule base and aggregate raw action scores.

        Fuzzy AND is implemented as min(...) of the antecedent degrees
        (standard Mamdani conjunction). Each rule's firing strength is
        scaled by a hand-tuned rule weight and accumulated (summed)
        into the relevant action's raw score. This produces a crisp,
        additive weighted-scoring system rather than a full centroid
        defuzzification, as specified.
        """
        weights: Dict[ActionType, float] = {
            ActionType.MOVE: self.BASELINE_WEIGHT,
            ActionType.EAT: self.BASELINE_WEIGHT,
            ActionType.ATTACK: self.BASELINE_WEIGHT,
            ActionType.COOPERATE: self.BASELINE_WEIGHT,
        }

        g_t = self.g_t
        g_e = self.g_e

        # R1: Starving + stressed + aggressive genotype -> strong ATTACK
        #     drive (desperate, violent resource competition).
        r1 = min(h_high, s_high, g_t)
        weights[ActionType.ATTACK] += r1 * 2.0

        # R2: Starving + aggressive genotype + a rival is actually nearby
        #     -> predatory ATTACK targeting that rival's resources.
        if rival_present:
            r2 = min(h_high, g_t)
            weights[ActionType.ATTACK] += r2 * 1.5

        # R3: High stress + aggressive genotype -> irritable ATTACK,
        #     independent of hunger (short-temper effect).
        r3 = min(s_high, g_t)
        weights[ActionType.ATTACK] += r3 * 1.0

        # R4: Starving + food is actually visible on the tile -> EAT.
        if food_present:
            r4 = h_high
            weights[ActionType.EAT] += r4 * 2.0
            r4b = min(h_med, g_e)
            weights[ActionType.EAT] += r4b * 0.75

        # R5: Starving + cooperative genotype -> COOPERATE (seek shared
        #     food / group foraging assistance), amplified further when
        #     no food is directly visible (must rely on the group).
        r5 = min(h_high, g_e)
        weights[ActionType.COOPERATE] += r5 * 1.25
        if not food_present:
            weights[ActionType.COOPERATE] += r5 * 0.5

        # R6: High stress + cooperative genotype -> COOPERATE (social
        #     bonding/grooming reduces stress in Eros-dominant agents).
        r6 = min(s_high, g_e)
        weights[ActionType.COOPERATE] += r6 * 1.0

        # R7: Low hunger + low stress -> MOVE/EXPLORE (comfortable,
        #     safe agents range further to map new territory).
        r7 = min(h_low, s_low)
        weights[ActionType.MOVE] += r7 * 1.5

        # R8: Medium hunger + low stress -> mild MOVE bias (searching
        #     for food proactively before starvation sets in).
        r8 = min(h_med, s_low)
        weights[ActionType.MOVE] += r8 * 1.0

        # R9: A rival is nearby + cooperative genotype + high stress ->
        #     COOPERATE (group defense / de-escalation instinct), while
        #     the same context with a moderately aggressive genotype
        #     nudges ATTACK instead (contested response to threat).
        if rival_present:
            r9 = min(s_high, g_e)
            weights[ActionType.COOPERATE] += r9 * 1.0
            r9b = min(g_t, s_med)
            weights[ActionType.ATTACK] += r9b * 0.75

        # R10: Medium hunger + aggressive genotype -> mild opportunistic
        #      ATTACK bias, much stronger when a rival is actually
        #      present to target.
        r10 = min(h_med, g_t) * (1.0 if rival_present else 0.3)
        weights[ActionType.ATTACK] += r10 * 0.5

        return weights

    def compute_action_weights(
        self, food_present: bool = False, rival_present: bool = False
    ) -> Dict[ActionType, float]:
        """Run the full fuzzy pipeline: fuzzify -> rules -> normalize.

        Args:
            food_present: Whether food currently occupies the agent's tile.
            rival_present: Whether a rival/competitor is within the
                            agent's local perception radius.

        Returns:
            A dict mapping each ActionType in [MOVE, EAT, ATTACK,
            COOPERATE] to a normalized preference/probability in
            [0.0, 1.0] that sums to 1.0. Side effect: caches fuzzified
            values and raw/normalized weights on `self` for logging.
        """
        h_low, h_med, h_high = self.fuzzify_hunger()
        s_low, s_med, s_high = self.fuzzify_stress()

        self.last_fuzzy_hunger = (h_low, h_med, h_high)
        self.last_fuzzy_stress = (s_low, s_med, s_high)

        raw_weights = self._evaluate_rule_base(
            h_low, h_med, h_high, s_low, s_med, s_high, food_present, rival_present
        )
        self.last_action_weights = raw_weights

        total = sum(raw_weights.values())
        if total <= 0.0:
            # Degenerate safety fallback: uniform distribution.
            n = len(raw_weights)
            probabilities = {action: 1.0 / n for action in raw_weights}
        else:
            probabilities = {action: value / total for action, value in raw_weights.items()}

        self.last_action_probabilities = probabilities
        return probabilities

    def decide_action(
        self,
        food_present: bool = False,
        rival_present: bool = False,
        deterministic: bool = False,
    ) -> Optional[ActionType]:
        """Select an action using the fuzzy emotional brain.

        Args:
            food_present: Whether food currently occupies the agent's tile.
            rival_present: Whether a rival is within perception range.
            deterministic: If True, greedily pick the highest-weighted
                            action. If False (default), sample
                            stochastically from the normalized action
                            distribution — reflecting the probabilistic,
                            "instinct under uncertainty" nature of the
                            emotional brain.

        Returns:
            The chosen ActionType, or None if the agent is not alive.
        """
        if not self.is_alive:
            self.last_chosen_action = None
            return None

        probabilities = self.compute_action_weights(food_present, rival_present)

        if deterministic:
            chosen = max(probabilities, key=probabilities.get)
        else:
            actions = list(probabilities.keys())
            weights = list(probabilities.values())
            chosen = self._rng.choices(actions, weights=weights, k=1)[0]

        self.last_chosen_action = chosen
        return chosen

    # ------------------------------------------------------------------
    # Bio-energetic metabolic engine
    # ------------------------------------------------------------------
    def apply_metabolism(self, action: ActionType) -> bool:
        """Apply the energy cost of `action` and refresh derived state.

        Decays `energy` according to ENERGY_COST, clamps it to [0, 100],
        recomputes `hunger` as (100 - energy), and transitions the agent
        to a dead state if energy has reached 0.

        Returns:
            True if the agent is still alive after this tick, False if
            this action killed it.
        """
        if not self.is_alive:
            return False

        cost = self.ENERGY_COST.get(action, 1.0)
        self.energy = max(0.0, min(100.0, self.energy - cost))
        self.hunger = 100.0 - self.energy

        if self.energy <= 0.0:
            self.is_alive = False

        return self.is_alive

    def update_stress(self, rival_count: int, smoothing: float = 0.3) -> float:
        """Recompute stress from local environmental context.

        Stress is driven by two components:
            1. Proximity to rivals/competitors: each nearby rival
               contributes up to 25 stress points, capped at 100.
            2. Starvation level: the agent's current hunger value is
               used directly as a stress driver (a starving agent is
               inherently more stressed).

        The two components are blended 50/50, and the result is
        exponentially smoothed against the previous stress value (using
        `smoothing` as the learning rate) to avoid unrealistic frame-to-
        frame stress spikes.

        Args:
            rival_count: Number of rival agents within perception range.
            smoothing: Blend factor in [0.0, 1.0] controlling how much
                       weight the new observation carries versus the
                       agent's previous stress level.

        Returns:
            The updated stress value.
        """
        rival_component = min(100.0, max(0, rival_count) * 25.0)
        starvation_component = self.hunger

        target_stress = 0.5 * rival_component + 0.5 * starvation_component
        target_stress = max(0.0, min(100.0, target_stress))

        smoothing = max(0.0, min(1.0, smoothing))
        self.stress = (1.0 - smoothing) * self.stress + smoothing * target_stress
        self.stress = max(0.0, min(100.0, self.stress))
        return self.stress

    def eat_single_food_item(self, food_energy_value: float) -> float:
        """Consume exactly ONE unit of food, restoring energy.

        This enforces the Single-Item Consumption constraint: the
        caller (typically the environment/ecosystem layer) is
        responsible for removing exactly one FoodItem from the tile
        (e.g. via `CongoEcosystem.consume_food_at(x, y, max_items=1)`)
        and passing its `energy` value in here. Any remaining food on
        the same tile is left untouched for other agents.

        Args:
            food_energy_value: The energy value of the single food item
                                consumed (e.g. 10 for North Bank food,
                                40 for South Bank food).

        Returns:
            The actual amount of energy gained (may be less than
            `food_energy_value` if the agent was already near full
            energy and the gain was clamped).
        """
        if not self.is_alive:
            return 0.0

        previous_energy = self.energy
        self.energy = max(0.0, min(100.0, self.energy + food_energy_value))
        self.hunger = 100.0 - self.energy
        return self.energy - previous_energy

    # ------------------------------------------------------------------
    # Convenience / introspection
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        """Return a plain-dict snapshot of the agent's current state."""
        return {
            "agent_id": self.agent_id,
            "position": (self.x, self.y),
            "generation": self.generation,
            "age": self.age,
            "max_age": self.max_age,
            "g_e": self.g_e,
            "g_t": self.g_t,
            "energy": self.energy,
            "hunger": self.hunger,
            "stress": self.stress,
            "is_alive": self.is_alive,
            "last_chosen_action": (
                self.last_chosen_action.value if self.last_chosen_action else None
            ),
        }

    def __repr__(self) -> str:
        return (
            f"EvolvableAgent(id={self.agent_id}, gen={self.generation}, "
            f"age={self.age}/{self.max_age}, "
            f"pos=({self.x},{self.y}), "
            f"G_E={self.g_e:.2f}, G_T={self.g_t:.2f}, energy={self.energy:.1f}, "
            f"hunger={self.hunger:.1f}, stress={self.stress:.1f}, "
            f"alive={self.is_alive})"
        )
