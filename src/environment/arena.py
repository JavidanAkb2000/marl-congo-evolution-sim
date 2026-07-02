"""
src/environment/arena.py

Phase 4: Multi-Agent Simulation & Training (The MARL Loop).

CongoArena is the top-level orchestrator that ties together:
    - CongoEcosystem (Phase 2): the 2D grid, river barrier, food spawning.
    - EvolvableAgent (Phase 3): individual agents with genes, metabolism,
      and a fuzzy-logic "emotional brain" for action selection.
    - InteractionResolver (Phase 4, System 1): the social collision
      matrix resolving same-cell agent encounters.
    - GeneticEngine (Phase 4, System 2): fertility, mating, crossover,
      and mutation.

API note: CongoArena mirrors the PettingZoo ParallelEnv surface
(`possible_agents`, `agents`, `reset()`, `step()`, `observe()`,
`rewards`/`terminations`/`truncations`/`infos`) so it can be dropped
into a standard PettingZoo-style training loop. It does not import the
`pettingzoo` package itself (no hard dependency is required to satisfy
that interface), so it will run in any environment without extra
installs; if you have `pettingzoo`/`gymnasium` installed, wrapping this
class in a formal `ParallelEnv` subclass is a thin, mechanical exercise
since the method names and semantics already match.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from src.agents.agent import ActionType, EvolvableAgent
from src.agents.genetics import GeneticEngine
from src.environment.ecosystem import CongoEcosystem
from src.environment.interactions import InteractionResolver

# The full discrete action space exposed to external policies. Note that
# the fuzzy brain itself only ever *chooses* from [MOVE, EAT, ATTACK,
# COOPERATE] (see Phase 3), but IDLE remains available so an external
# RL policy overriding an agent's action has a genuine "do nothing" option.
ACTION_SPACE: Tuple[ActionType, ...] = (
    ActionType.IDLE,
    ActionType.MOVE,
    ActionType.EAT,
    ActionType.ATTACK,
    ActionType.COOPERATE,
)


class CongoArena:
    """The Phase 4 multi-agent environment: the full MARL simulation loop."""

    metadata = {"name": "congo_river_paradigm_v4", "render_modes": ["human", "ansi"]}

    def __init__(
        self,
        ecosystem: CongoEcosystem,
        initial_population: int = 20,
        perception_radius: int = 3,
        genetic_engine: Optional[GeneticEngine] = None,
        interaction_resolver: Optional[InteractionResolver] = None,
        max_steps: Optional[int] = None,
        rng_seed: Optional[int] = None,
    ) -> None:
        """Construct a CongoArena.

        Args:
            ecosystem: A configured (but not necessarily reset) CongoEcosystem.
            initial_population: Number of agents spawned by reset().
            perception_radius: Chebyshev radius within which an agent can
                                perceive rivals for its fuzzy brain's
                                `rival_present` sensory input.
            genetic_engine: Optional custom GeneticEngine. A default is
                             constructed if omitted.
            interaction_resolver: Optional custom InteractionResolver. A
                                   default is constructed if omitted.
            max_steps: Optional episode length; when current_step reaches
                       this value, all agents are marked truncated.
            rng_seed: Seed for the arena's shared RNG (drives spawning,
                      movement, combat rolls, and mutation).
        """
        self.ecosystem = ecosystem
        self.perception_radius = perception_radius
        self.max_steps = max_steps

        self._rng = random.Random(rng_seed)
        self.genetic_engine = genetic_engine or GeneticEngine(rng=self._rng)
        self.interaction_resolver = interaction_resolver or InteractionResolver(rng=self._rng)

        self._initial_population = initial_population
        self._next_agent_id: int = 0

        self.agents_by_id: Dict[int, EvolvableAgent] = {}
        self.possible_agents: List[str] = []
        self.agents: List[str] = []

        self.current_step: int = 0

        self.rewards: Dict[str, float] = {}
        self.terminations: Dict[str, bool] = {}
        self.truncations: Dict[str, bool] = {}
        self.infos: Dict[str, dict] = {}

        # Human-readable log of everything that happened in the most
        # recent step() call (interactions + reproductions), useful for
        # debugging, telemetry, or driving a visualization layer.
        self.last_step_log: List[str] = []

    # ------------------------------------------------------------------
    # Agent naming helpers (PettingZoo convention: string agent names)
    # ------------------------------------------------------------------
    @staticmethod
    def _agent_name(agent_id: int) -> str:
        return f"agent_{agent_id}"

    @staticmethod
    def _agent_id_from_name(name: str) -> int:
        return int(name.rsplit("_", 1)[1])

    # ------------------------------------------------------------------
    # Spaces (lightweight, dependency-free stand-ins for gym.Space)
    # ------------------------------------------------------------------
    def observation_space(self, agent_name: str) -> Dict[str, object]:
        """Describe the observation layout for a given agent.

        Returned as a plain descriptor dict (no hard dependency on
        gymnasium/pettingzoo). Swap this for a real
        `gymnasium.spaces.Dict` in a training harness if desired — the
        keys and bounds below map directly onto one.
        """
        return {
            "x": {"low": 0, "high": self.ecosystem.width - 1, "dtype": "int"},
            "y": {"low": 0, "high": self.ecosystem.height - 1, "dtype": "int"},
            "energy": {"low": 0.0, "high": 100.0, "dtype": "float"},
            "hunger": {"low": 0.0, "high": 100.0, "dtype": "float"},
            "stress": {"low": 0.0, "high": 100.0, "dtype": "float"},
            "g_e": {"low": 0.0, "high": 1.0, "dtype": "float"},
            "g_t": {"low": 0.0, "high": 1.0, "dtype": "float"},
            "generation": {"low": 0, "high": float("inf"), "dtype": "int"},
            "food_present": {"values": [False, True], "dtype": "bool"},
            "rival_present": {"values": [False, True], "dtype": "bool"},
            "rival_count": {"low": 0, "high": float("inf"), "dtype": "int"},
            "bank": {"values": ["north", "south"], "dtype": "categorical"},
        }

    def action_space(self, agent_name: str) -> Tuple[ActionType, ...]:
        """Return the discrete action space available to any agent."""
        return ACTION_SPACE

    # ------------------------------------------------------------------
    # Population management
    # ------------------------------------------------------------------
    def _spawn_agent(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        forced_g_t: Optional[float] = None,
        generation: int = 0,
        initial_energy: float = 100.0,
    ) -> EvolvableAgent:
        agent_id = self._next_agent_id
        self._next_agent_id += 1

        if x is None:
            x = self._rng.randint(0, self.ecosystem.width - 1)
        if y is None:
            y = self._rng.randint(0, self.ecosystem.height - 1)

        agent = EvolvableAgent(
            agent_id=agent_id,
            x=x,
            y=y,
            forced_g_t=forced_g_t,
            initial_energy=initial_energy,
            generation=generation,
            rng=self._rng,
        )
        self.agents_by_id[agent_id] = agent
        name = self._agent_name(agent_id)
        if name not in self.possible_agents:
            self.possible_agents.append(name)
        return agent

    def reset(self, seed: Optional[int] = None) -> Dict[str, dict]:
        """Reset the ecosystem and spawn a fresh initial population.

        Returns:
            A dict mapping each alive agent's name to its observation.
        """
        if seed is not None:
            self._rng.seed(seed)

        self.ecosystem.reset()
        self.agents_by_id.clear()
        self.possible_agents = []
        self._next_agent_id = 0
        self.current_step = 0
        self.last_step_log = []

        for _ in range(self._initial_population):
            self._spawn_agent()

        self.agents = [self._agent_name(aid) for aid in self.agents_by_id]
        self.rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        return {
            name: self._get_observation(self.agents_by_id[self._agent_id_from_name(name)])
            for name in self.agents
        }

    # ------------------------------------------------------------------
    # Sensing / observations
    # ------------------------------------------------------------------
    def _sense(self, agent: EvolvableAgent) -> Tuple[bool, bool, int]:
        """Return (food_present, rival_present, rival_count) for an agent."""
        food_present = len(self.ecosystem.get_food_at(agent.x, agent.y)) > 0

        rival_count = 0
        for other in self.agents_by_id.values():
            if other.agent_id == agent.agent_id or not other.is_alive:
                continue
            distance = max(abs(other.x - agent.x), abs(other.y - agent.y))
            if distance <= self.perception_radius:
                rival_count += 1

        return food_present, rival_count > 0, rival_count

    def _get_observation(self, agent: EvolvableAgent) -> dict:
        food_present, rival_present, rival_count = self._sense(agent)
        return {
            "x": agent.x,
            "y": agent.y,
            "energy": agent.energy,
            "hunger": agent.hunger,
            "stress": agent.stress,
            "g_e": agent.g_e,
            "g_t": agent.g_t,
            "generation": agent.generation,
            "food_present": food_present,
            "rival_present": rival_present,
            "rival_count": rival_count,
            "bank": self.ecosystem.get_bank(agent.y),
        }

    def observe(self, agent_name: str) -> dict:
        """PettingZoo-style single-agent observation accessor."""
        agent = self.agents_by_id.get(self._agent_id_from_name(agent_name))
        if agent is None or not agent.is_alive:
            return {}
        return self._get_observation(agent)

    # ------------------------------------------------------------------
    # The main step function
    # ------------------------------------------------------------------
    def step(
        self, actions: Optional[Dict[str, ActionType]] = None
    ) -> Tuple[Dict[str, dict], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]]:
        """Advance the simulation by one full tick.

        Pipeline:
            1. Decision phase: each living agent's action is either
               taken from `actions` (external policy override) or
               computed via its own fuzzy emotional brain.
            2. Metabolism phase: apply the base energy cost of each
               chosen action.
            3. Interaction phase: resolve the social collision matrix
               for every cell containing 2+ agents.
            4. Individual resolution phase: agents whose action was not
               fully consumed by an interaction execute it normally
               (movement / eating).
            5. Ecosystem tick: spawn new food across both banks.
            6. Reproduction phase: run the genetic algorithm over all
               currently fertile, compatible, non-aggressive agents.
            7. Bookkeeping: build PettingZoo-style
               observations/rewards/terminations/truncations/infos.

        Args:
            actions: Optional external action overrides keyed by agent
                     name. Any agent omitted from this dict falls back
                     to its own fuzzy brain's decision.

        Returns:
            (observations, rewards, terminations, truncations, infos)
        """
        actions = actions or {}
        log: List[str] = []

        living_ids = [aid for aid, a in self.agents_by_id.items() if a.is_alive]
        chosen: Dict[int, ActionType] = {}
        pre_step_energy: Dict[int, float] = {}

        # --- 1. Decision phase ---
        for aid in living_ids:
            agent = self.agents_by_id[aid]
            pre_step_energy[aid] = agent.energy
            name = self._agent_name(aid)

            override = actions.get(name)
            if override is not None:
                chosen[aid] = override
            else:
                food_present, rival_present, _ = self._sense(agent)
                chosen[aid] = agent.decide_action(
                    food_present=food_present, rival_present=rival_present
                )

        # --- 2. Metabolism phase ---
        for aid in living_ids:
            agent = self.agents_by_id[aid]
            if agent.is_alive:
                agent.apply_metabolism(chosen[aid])

        # --- 3. Interaction phase (collision matrix) ---
        cells: Dict[Tuple[int, int], List[int]] = {}
        for aid in living_ids:
            agent = self.agents_by_id[aid]
            if not agent.is_alive:
                continue
            cells.setdefault((agent.x, agent.y), []).append(aid)

        suppress: Dict[int, bool] = {aid: False for aid in living_ids}
        for occupant_ids in cells.values():
            if len(occupant_ids) < 2:
                continue
            occupant_agents = [
                self.agents_by_id[oid] for oid in occupant_ids if self.agents_by_id[oid].is_alive
            ]
            pair_suppress, pair_log = self.interaction_resolver.resolve_cell(
                occupant_agents, chosen, self.ecosystem
            )
            suppress.update(pair_suppress)
            log.extend(pair_log)

        # --- 4. Individual resolution phase ---
        for aid in living_ids:
            agent = self.agents_by_id[aid]
            if not agent.is_alive or suppress.get(aid, False):
                continue
            self._apply_individual_action(agent, chosen[aid])

        # --- 5. Ecosystem tick ---
        self.ecosystem.step()

        # --- 6. Reproduction phase ---
        log.extend(self._process_reproduction(chosen))

        # --- 7. Bookkeeping ---
        self.current_step += 1
        self.last_step_log = log

        observations: Dict[str, dict] = {}
        rewards: Dict[str, float] = {}
        terminations: Dict[str, bool] = {}
        truncations: Dict[str, bool] = {}
        infos: Dict[str, dict] = {}
        alive_names: List[str] = []

        truncated_episode = bool(self.max_steps) and self.current_step >= self.max_steps

        for aid, agent in list(self.agents_by_id.items()):
            name = self._agent_name(aid)
            terminated = not agent.is_alive
            reward = agent.energy - pre_step_energy.get(aid, agent.energy)

            observations[name] = self._get_observation(agent) if agent.is_alive else {}
            rewards[name] = reward
            terminations[name] = terminated
            truncations[name] = truncated_episode
            infos[name] = {
                "action": chosen[aid].value if aid in chosen and chosen[aid] else None,
                **agent.to_dict(),
            }
            if agent.is_alive:
                alive_names.append(name)

        self.agents = alive_names
        self.rewards, self.terminations, self.truncations, self.infos = (
            rewards,
            terminations,
            truncations,
            infos,
        )

        # Prune agents that died this step from the live population dict.
        # Their final state was already captured above in `infos`, so it
        # is safe to drop them now; this keeps `agents_by_id` bounded to
        # the currently-living population over long-running simulations
        # instead of growing without limit. `possible_agents` (the
        # historical name registry) is left untouched.
        for aid in list(self.agents_by_id.keys()):
            if not self.agents_by_id[aid].is_alive:
                del self.agents_by_id[aid]

        return observations, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    # Individual action resolution
    # ------------------------------------------------------------------
    def _apply_individual_action(self, agent: EvolvableAgent, action: ActionType) -> None:
        if action == ActionType.MOVE:
            self._move_agent(agent)
        elif action == ActionType.EAT:
            energy_value = self.ecosystem.consume_food_at(agent.x, agent.y, max_items=1)
            if energy_value > 0.0:
                agent.eat_single_food_item(energy_value)
        elif action in (ActionType.ATTACK, ActionType.COOPERATE):
            # No other agent was present to interact with (a solo
            # ATTACK/COOPERATE): the metabolic cost already applied in
            # phase 2 is the only consequence.
            pass
        # ActionType.IDLE: no-op by definition.

    def _move_agent(self, agent: EvolvableAgent) -> None:
        """Move an agent one cell in a random legal direction (8-connected)."""
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        self._rng.shuffle(offsets)

        for dx, dy in offsets:
            target_x, target_y = agent.x + dx, agent.y + dy
            if not self.ecosystem.is_within_bounds(target_x, target_y):
                continue
            if not self.ecosystem.is_move_legal(agent.y, target_y):
                continue
            agent.x, agent.y = target_x, target_y
            return
        # Agent is fully boxed in by boundaries; it stays in place.

    # ------------------------------------------------------------------
    # Reproduction / genetic algorithm phase
    # ------------------------------------------------------------------
    def _process_reproduction(self, chosen_actions: Dict[int, ActionType]) -> List[str]:
        log: List[str] = []

        living = [a for a in self.agents_by_id.values() if a.is_alive]
        fertile_sorted = sorted(
            (a for a in living if self.genetic_engine.is_fertile(a)),
            key=lambda a: a.agent_id,
        )

        used_ids: set = set()

        for i, parent1 in enumerate(fertile_sorted):
            if parent1.agent_id in used_ids or not parent1.is_alive:
                continue

            for parent2 in fertile_sorted[i + 1 :]:
                if parent2.agent_id in used_ids or not parent2.is_alive:
                    continue

                action1 = chosen_actions.get(parent1.agent_id)
                action2 = chosen_actions.get(parent2.agent_id)

                if not self.genetic_engine.can_mate(parent1, parent2, action1, action2):
                    continue

                child_position = (parent1.x, parent1.y)
                child = self.genetic_engine.reproduce(
                    parent1, parent2, self._next_agent_id, child_position
                )
                self._next_agent_id += 1

                self.agents_by_id[child.agent_id] = child
                self.possible_agents.append(self._agent_name(child.agent_id))

                self.genetic_engine.apply_reproduction_cost(parent1)
                self.genetic_engine.apply_reproduction_cost(parent2)

                used_ids.add(parent1.agent_id)
                used_ids.add(parent2.agent_id)

                log.append(
                    f"REPRODUCTION: Agent {parent1.agent_id} x Agent {parent2.agent_id} "
                    f"-> offspring Agent {child.agent_id} "
                    f"(generation {child.generation}, G_T={child.g_t:.3f}, G_E={child.g_e:.3f})"
                )
                break  # parent1 has mated this step; move to the next candidate.

        return log

    # ------------------------------------------------------------------
    # Rendering / lifecycle
    # ------------------------------------------------------------------
    def render(self, mode: str = "human") -> Optional[str]:
        """Render a lightweight textual summary of the current world state."""
        alive_agents = [a for a in self.agents_by_id.values() if a.is_alive]
        north_count = sum(1 for a in alive_agents if self.ecosystem.get_bank(a.y) == "north")
        south_count = len(alive_agents) - north_count
        avg_g_t = (
            sum(a.g_t for a in alive_agents) / len(alive_agents) if alive_agents else 0.0
        )

        summary = (
            f"[Step {self.current_step}] "
            f"Population: {len(alive_agents)} "
            f"(North={north_count}, South={south_count}) | "
            f"Avg G_T={avg_g_t:.3f} | "
            f"Food on grid: {self.ecosystem.food_count()}"
        )

        if mode == "ansi":
            return summary

        print(summary)
        for line in self.last_step_log:
            print(f"    {line}")
        return None

    def close(self) -> None:
        """No external resources are held; provided for API parity."""
        return None