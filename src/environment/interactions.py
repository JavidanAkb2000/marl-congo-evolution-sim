"""
src/environment/interactions.py

Phase 4, System 1: Social Interactions (The Collision Matrix).

Whenever two or more agents occupy the exact same grid coordinate, this
module intercepts their independently-chosen fuzzy actions and resolves
the social consequences of that collision (combat, predation, alliance)
before the environment applies each agent's action individually.

This module has no knowledge of the fuzzy brain itself; it only
consumes the already-chosen ActionType per agent and mutates agent
state (energy, hunger, stress, position) as a result of the encounter.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from src.agents.agent import ActionType, EvolvableAgent
from src.environment.ecosystem import CongoEcosystem


class InteractionResolver:
    """Resolves same-cell multi-agent collisions ("the collision matrix").

    Design notes / explicit assumptions (the spec leaves these details
    to implementation judgment):
      - Interactions are resolved strictly for agents sharing the exact
        same (x, y) coordinate. Adjacency-based encounters are handled
        separately by the reproduction system, not here.
      - If more than two agents share a cell, agents are paired off
        sequentially (in randomized order) and each pair is resolved
        independently; an odd agent out simply proceeds with its
        chosen action individually (no interaction).
      - "Defenseless while EATing" is treated as its own case
        (ATTACK vs EAT), distinct from ATTACK vs COOPERATE, and doubles
        the base damage the eating agent takes while also interrupting
        the meal (no food is consumed that step).
      - Any action pairing not covered by an explicit rule (e.g.
        MOVE vs EAT, IDLE vs COOPERATE) is treated as a non-conflicting
        co-location: both agents proceed with their own action
        individually, unaffected by each other.
    """

    # Tunable interaction constants.
    #
    # === REBALANCED FOR ECOLOGICAL STABILITY ===
    # The original values (COMBAT_ENERGY_LOSS=20, PREDATION_BASE_DAMAGE=15)
    # were severe enough that even a single fight could wipe out most of
    # an agent's energy bar outright, and a starving/aggressive population
    # would chain ATTACK actions step after step, guaranteeing a rapid
    # extinction death-spiral regardless of food availability. These
    # values are lowered so combat is still meaningfully costly and
    # dangerous, but survivable for a healthy agent, and no longer
    # single-handedly drives total population collapse.
    COMBAT_ENERGY_LOSS: float = 5.0
    COMBAT_SPOILS: float = 6.0
    COMBAT_STRESS_GAIN: float = 10.0

    PREDATION_BASE_DAMAGE: float = 8.0
    PREDATION_STOLEN_ENERGY: float = 6.0
    PREDATION_VICTIM_STRESS_GAIN: float = 15.0
    PREDATION_ATTACKER_STRESS_GAIN: float = 3.0

    # Being caught mid-EAT is still a real, elevated risk (that's the
    # point of the rule), but no longer an almost-guaranteed kill: on
    # the old constants a full-health victim could lose 30 damage + 15
    # stolen = 45 energy in one hit; now it's a survivable-but-costly
    # 12 damage + 8 stolen = 20 energy.
    DEFENSELESS_DAMAGE_MULTIPLIER: float = 1.5
    DEFENSELESS_STOLEN_ENERGY: float = 8.0
    DEFENSELESS_VICTIM_STRESS_GAIN: float = 20.0

    # Cooperation remains clearly the "safer" strategy: a strong stress
    # payoff keeps peaceful agents comparatively calmer than combatants.
    ALLIANCE_STRESS_RELIEF: float = 25.0

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self._rng = rng if rng is not None else random.Random()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def resolve_cell(
        self,
        occupants: List[EvolvableAgent],
        chosen_actions: Dict[int, ActionType],
        ecosystem: CongoEcosystem,
    ) -> Tuple[Dict[int, bool], List[str]]:
        """Resolve all interactions for agents sharing a single cell.

        Args:
            occupants: All living agents currently at this coordinate.
            chosen_actions: Map of agent_id -> the ActionType each agent
                             independently chose this step (from the
                             fuzzy brain or an external policy).
            ecosystem: The shared ecosystem, needed for cooperative
                       food-splitting and territory-aware repel moves.

        Returns:
            A tuple (suppress, log):
              - suppress: agent_id -> True if this agent's action was
                fully resolved by an interaction and should NOT also be
                processed individually afterwards (e.g. it already ate
                its share, or its attack already applied its damage).
              - log: human-readable strings describing what happened,
                useful for debugging/telemetry.
        """
        suppress: Dict[int, bool] = {a.agent_id: False for a in occupants}
        log: List[str] = []

        if len(occupants) < 2:
            return suppress, log

        shuffled = list(occupants)
        self._rng.shuffle(shuffled)

        for i in range(0, len(shuffled) - 1, 2):
            a1, a2 = shuffled[i], shuffled[i + 1]
            if not (a1.is_alive and a2.is_alive):
                continue

            act1 = chosen_actions.get(a1.agent_id, ActionType.IDLE)
            act2 = chosen_actions.get(a2.agent_id, ActionType.IDLE)

            pair_suppress, pair_log = self._resolve_pair(a1, act1, a2, act2, ecosystem)
            suppress.update(pair_suppress)
            log.extend(pair_log)

        # An odd agent left unpaired proceeds individually; its
        # suppress flag is already False from initialization above.
        return suppress, log

    # ------------------------------------------------------------------
    # Pairwise resolution
    # ------------------------------------------------------------------
    def _resolve_pair(
        self,
        a1: EvolvableAgent,
        act1: ActionType,
        a2: EvolvableAgent,
        act2: ActionType,
        ecosystem: CongoEcosystem,
    ) -> Tuple[Dict[int, bool], List[str]]:
        actions = {act1, act2}

        if act1 == ActionType.ATTACK and act2 == ActionType.ATTACK:
            return self._resolve_combat(a1, a2, ecosystem)

        if ActionType.ATTACK in actions and ActionType.EAT in actions:
            attacker, victim = (a1, a2) if act1 == ActionType.ATTACK else (a2, a1)
            return self._resolve_predation(attacker, victim, defenseless=True)

        if ActionType.ATTACK in actions and ActionType.COOPERATE in actions:
            attacker, victim = (a1, a2) if act1 == ActionType.ATTACK else (a2, a1)
            return self._resolve_predation(attacker, victim, defenseless=False)

        if ActionType.ATTACK in actions:
            # Attacker vs a MOVE/IDLE bystander: opportunistic, non-
            # defenseless predation (same base severity as vs COOPERATE).
            attacker, victim = (a1, a2) if act1 == ActionType.ATTACK else (a2, a1)
            return self._resolve_predation(attacker, victim, defenseless=False)

        if act1 == ActionType.COOPERATE and act2 == ActionType.COOPERATE:
            return self._resolve_alliance(a1, a2, ecosystem)

        # No conflict rule applies: both agents proceed independently.
        suppress = {a1.agent_id: False, a2.agent_id: False}
        return suppress, []

    # ------------------------------------------------------------------
    # Individual rule implementations
    # ------------------------------------------------------------------
    def _resolve_combat(
        self, a1: EvolvableAgent, a2: EvolvableAgent, ecosystem: CongoEcosystem
    ) -> Tuple[Dict[int, bool], List[str]]:
        """ATTACK vs ATTACK: severe mutual combat."""
        self._apply_damage(a1, self.COMBAT_ENERGY_LOSS)
        self._apply_damage(a2, self.COMBAT_ENERGY_LOSS)

        # Combat strength combines aggression (g_t) and body size
        # (g_size): a bigger, more aggressive agent wins more often. Using
        # the product means a heavy chimp has a real edge over a light
        # bonobo of equal aggression, reflecting mass advantage in a fight.
        strength_a1 = a1.g_t * a1.g_size
        strength_a2 = a2.g_t * a2.g_size
        strength_sum = strength_a1 + strength_a2
        win_prob_a1 = strength_a1 / strength_sum if strength_sum > 0 else 0.5
        winner, loser = (a1, a2) if self._rng.random() < win_prob_a1 else (a2, a1)

        log: List[str] = [
            f"COMBAT: Agent {a1.agent_id} (G_T={a1.g_t:.2f}) vs "
            f"Agent {a2.agent_id} (G_T={a2.g_t:.2f}) -> winner: Agent {winner.agent_id}"
        ]

        if winner.is_alive and loser.is_alive:
            spoils = min(self.COMBAT_SPOILS, loser.energy)
            self._apply_damage(loser, spoils)
            winner.energy = max(0.0, min(100.0, winner.energy + spoils))
            winner.hunger = 100.0 - winner.energy
            log.append(
                f"COMBAT: Agent {winner.agent_id} claims {spoils:.1f} energy "
                f"in spoils from Agent {loser.agent_id}."
            )

        for combatant in (a1, a2):
            combatant.stress = max(0.0, min(100.0, combatant.stress + self.COMBAT_STRESS_GAIN))

        if loser.is_alive:
            self._repel(loser, ecosystem)
            log.append(f"COMBAT: Agent {loser.agent_id} is repelled from the contested cell.")
        else:
            log.append(f"COMBAT: Agent {loser.agent_id} did not survive the encounter.")

        return {a1.agent_id: True, a2.agent_id: True}, log

    def _resolve_predation(
        self, attacker: EvolvableAgent, victim: EvolvableAgent, defenseless: bool
    ) -> Tuple[Dict[int, bool], List[str]]:
        """ATTACK vs COOPERATE/EAT/MOVE/IDLE: one-sided predation.

        If `defenseless` is True (the victim was caught mid-EAT), the
        base damage is doubled and the victim's meal is interrupted
        (no food is consumed by the victim this step).
        """
        multiplier = self.DEFENSELESS_DAMAGE_MULTIPLIER if defenseless else 1.0
        damage = self.PREDATION_BASE_DAMAGE * multiplier
        stolen = min(
            self.DEFENSELESS_STOLEN_ENERGY if defenseless else self.PREDATION_STOLEN_ENERGY,
            max(0.0, victim.energy - damage) if victim.energy > damage else 0.0,
        )

        self._apply_damage(victim, damage)
        if stolen > 0.0:
            # Stolen energy is transferred on top of the direct damage
            # already applied (representing looted food/resources).
            self._apply_damage(victim, stolen)
            attacker.energy = max(0.0, min(100.0, attacker.energy + stolen))
            attacker.hunger = 100.0 - attacker.energy

        victim_stress_gain = (
            self.DEFENSELESS_VICTIM_STRESS_GAIN if defenseless else self.PREDATION_VICTIM_STRESS_GAIN
        )
        victim.stress = max(0.0, min(100.0, victim.stress + victim_stress_gain))
        attacker.stress = max(
            0.0, min(100.0, attacker.stress + self.PREDATION_ATTACKER_STRESS_GAIN)
        )

        descriptor = "defenseless (caught mid-EAT)" if defenseless else "predation"
        log = [
            f"ATTACK: Agent {attacker.agent_id} preys on Agent {victim.agent_id} "
            f"({descriptor}); damage={damage:.1f}, stolen={stolen:.1f}."
        ]
        if not victim.is_alive:
            log.append(f"ATTACK: Agent {victim.agent_id} did not survive the attack.")

        # Both actions are fully resolved by this interaction: the
        # attacker's ATTACK already landed, and the victim's original
        # action (COOPERATE/EAT/MOVE/IDLE) is interrupted by the assault.
        return {attacker.agent_id: True, victim.agent_id: True}, log

    def _resolve_alliance(
        self, a1: EvolvableAgent, a2: EvolvableAgent, ecosystem: CongoEcosystem
    ) -> Tuple[Dict[int, bool], List[str]]:
        """COOPERATE vs COOPERATE: peaceful alliance, food split equally."""
        log: List[str] = [f"ALLIANCE: Agent {a1.agent_id} and Agent {a2.agent_id} cooperate."]

        food_here = ecosystem.get_food_at(a1.x, a1.y)
        if food_here:
            total_energy = ecosystem.consume_food_at(a1.x, a1.y, max_items=1)
            if total_energy > 0.0:
                half_share = total_energy / 2.0
                gained_1 = a1.eat_single_food_item(half_share)
                gained_2 = a2.eat_single_food_item(half_share)
                log.append(
                    f"ALLIANCE: Food item (value={total_energy:.1f}) split equally "
                    f"-> Agent {a1.agent_id} +{gained_1:.1f}, Agent {a2.agent_id} +{gained_2:.1f}."
                )

        for ally in (a1, a2):
            ally.stress = max(0.0, min(100.0, ally.stress - self.ALLIANCE_STRESS_RELIEF))

        return {a1.agent_id: True, a2.agent_id: True}, log

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_damage(agent: EvolvableAgent, amount: float) -> None:
        """Directly reduce an agent's energy (outside the normal
        metabolic cost model), recompute hunger, and flag death."""
        if amount <= 0.0 or not agent.is_alive:
            return
        agent.energy = max(0.0, min(100.0, agent.energy - amount))
        agent.hunger = 100.0 - agent.energy
        if agent.energy <= 0.0:
            agent.is_alive = False

    def _repel(self, agent: EvolvableAgent, ecosystem: CongoEcosystem) -> None:
        """Push a combat loser into a random adjacent legal cell.

        Legality means: within the hard grid boundary AND on the same
        bank of the Congo River (repelling can never fling an agent
        across the river barrier).
        """
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        self._rng.shuffle(offsets)

        for dx, dy in offsets:
            target_x, target_y = agent.x + dx, agent.y + dy
            if not ecosystem.is_within_bounds(target_x, target_y):
                continue
            if not ecosystem.is_move_legal(agent.y, target_y):
                continue
            agent.x, agent.y = target_x, target_y
            return
        # No legal adjacent cell found (fully boxed in) -> agent stays put.