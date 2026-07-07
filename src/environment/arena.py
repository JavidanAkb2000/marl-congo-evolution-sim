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
        foraging_radius: Optional[int] = None,
        food_seeking_bias: float = 0.85,
        foraging_override_hunger_threshold: float = 40.0,
        north_population_ratio: float = 0.70,
        north_max_age_range: Tuple[int, int] = (150, 250),
        south_max_age_range: Optional[Tuple[int, int]] = None,
        north_clan_spawn_radius: int = 6,
        initial_north_population: Optional[int] = None,
        initial_south_population: Optional[int] = None,
        gorilla_stress_penalty: float = 25.0,
        gorilla_energy_penalty: float = 4.0,
        north_birth_dispersal_radius: int = 8,
        crowding_radius: int = 3,
        crowding_soft_cap: int = 8,
        crowding_stress_per_excess: float = 1.5,
        crowding_migration_trigger: int = 12,
        migration_vision_radius: int = 25,
        north_gene_means: Tuple[float, float] = (1.2, 0.85),
        south_gene_means: Tuple[float, float] = (0.85, 0.30),
        north_reproduction_cooldown: int = 20,
        south_reproduction_cooldown: int = 60,
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
            foraging_radius: Chebyshev radius within which an agent can
                              *see* food and steer toward it while
                              executing a MOVE action. Defaults to
                              `perception_radius` if omitted, so the
                              same "vision" applies to both rival and
                              food awareness unless explicitly split.
            food_seeking_bias: Probability, in [0.0, 1.0], that a MOVE
                                action steps directly toward the nearest
                                visible food item rather than a fully
                                random step. Kept below 1.0 on purpose
                                so foraging isn't a perfect, deterministic
                                pathfinder — agents still occasionally
                                wander, preserving some exploration.
            foraging_override_hunger_threshold: Hunger level above which
                                the arena will override a "do-nothing"
                                fuzzy-brain choice (EAT/COOPERATE/ATTACK
                                attempted with no food or partner
                                actually present) with MOVE, provided
                                food is visible within `foraging_radius`.
                                This closes a real blind spot in the
                                Phase 3 fuzzy brain: its rules only favor
                                MOVE when hunger is LOW, so a starving
                                agent can get stuck repeatedly attempting
                                a no-op action in place while reachable
                                food sits untouched nearby. Set to a
                                value >= 100 to disable this override
                                entirely and restore pure fuzzy-brain
                                behavior.
            north_population_ratio: Fraction of `initial_population`
                                spawned on the North bank by reset()
                                (the remainder spawns on the South bank).
                                Modeling the "many small-bodied,
                                high-turnover Chimps vs fewer, stable
                                Bonobos" demographic profile: a larger
                                starting North cohort absorbs early
                                intra-species selection losses.
            north_max_age_range: Inclusive (min, max) lifespan cap, in
                                simulation ticks, randomly assigned to
                                each North-bank agent at birth (shorter
                                — high stress/combat wear-and-tear).
            south_max_age_range: Inclusive (min, max) lifespan cap for
                                each South-bank agent at birth (longer —
                                peaceful, stable life), or None to
                                disable aging-based death entirely for
                                South agents. Defaults to None: a
                                finite cap, however generous, still
                                turns a rare-but-survivable population
                                stall (e.g. down to 2 agents who
                                struggle to find each other to mate)
                                into a guaranteed extinction deadline
                                once both individuals age out. Since
                                South is meant to represent a stable,
                                low-conflict population, removing the
                                cap entirely is the more faithful
                                choice; set a finite range instead if
                                bounded lifespans are specifically
                                wanted for South.
            north_clan_spawn_radius: Chebyshev radius around a randomly
                                chosen North food hotspot within which
                                each initial North-bank agent spawns.
                                Models a troupe already living where its
                                food source is, rather than a Chimp
                                being placed uniformly at random
                                somewhere in the ~1200-cell North band
                                (most of which is desert) with no
                                realistic chance of ever reaching a
                                hotspot before starving.
            initial_north_population: If given TOGETHER WITH
                                `initial_south_population`, these two
                                explicit counts override the
                                `initial_population`/
                                `north_population_ratio` split entirely.
                                Useful for precise demographic
                                calibration (e.g. "80 North, 60 South")
                                without depending on rounding a ratio
                                against a combined total.
            initial_south_population: See `initial_north_population`.
        """
        self.ecosystem = ecosystem
        self.perception_radius = perception_radius
        self.foraging_radius = foraging_radius if foraging_radius is not None else perception_radius
        self.food_seeking_bias = max(0.0, min(1.0, food_seeking_bias))
        self.foraging_override_hunger_threshold = foraging_override_hunger_threshold
        self.north_population_ratio = max(0.0, min(1.0, north_population_ratio))
        self.north_max_age_range = north_max_age_range
        self.south_max_age_range = south_max_age_range
        self.north_clan_spawn_radius = north_clan_spawn_radius
        self.initial_north_population = initial_north_population
        self.initial_south_population = initial_south_population
        self.gorilla_stress_penalty = gorilla_stress_penalty
        self.gorilla_energy_penalty = gorilla_energy_penalty
        self.north_birth_dispersal_radius = north_birth_dispersal_radius
        self.crowding_radius = crowding_radius
        self.crowding_soft_cap = crowding_soft_cap
        self.crowding_stress_per_excess = crowding_stress_per_excess
        self.crowding_migration_trigger = crowding_migration_trigger
        self.migration_vision_radius = migration_vision_radius
        # Per-bank (g_size, g_fertility) gene means. North chimps are
        # heavier (costlier metabolism, stronger combat) and high-fertility
        # (fast breeders); South bonobos are lighter (cheap metabolism) and
        # low-fertility (slow, selective breeders). Individual agents are
        # drawn with small jitter around these means so selection has
        # variation to act on.
        self.north_gene_means = north_gene_means
        self.south_gene_means = south_gene_means
        # Per-bank inter-birth interval. North (chimp) uses a SHORT
        # cooldown: it's a high-mortality, r-selected "breed fast to
        # replace losses" strategy, and a long cooldown starves its
        # recovery so deaths outpace births. South (bonobo) uses a LONG
        # cooldown: a low-mortality, K-selected strategy where the long
        # interval damps booms toward a stable logistic equilibrium.
        # (Assigned per-parent by bank at birth-time, mirroring how
        # max_age is a bank/ecology concern rather than a genetics one.)
        self.north_reproduction_cooldown = max(0, int(north_reproduction_cooldown))
        self.south_reproduction_cooldown = max(0, int(south_reproduction_cooldown))

        # Per-step caches (populated during step(); initialized here so
        # any early access is safe).
        self._density_map: Dict[Tuple[int, int], int] = {}
        self._local_crowd_cache: Dict[int, int] = {}
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
        bank: Optional[str] = None,
    ) -> EvolvableAgent:
        """Create and register a new agent.

        If `bank == "north"` (and x/y aren't explicitly supplied), the
        agent spawns as part of a "clan/troupe": placed within
        `north_clan_spawn_radius` of a randomly-chosen North food
        hotspot, rather than scattered anywhere across the whole
        1200-cell North band. This is what actually makes the patchy
        food design viable — a Chimp born miles from every hotspot has
        no realistic way to ever reach one before starving, so the
        initial population needs to already live where the food is,
        the same way a real troupe occupies its territory around a
        food source rather than being airdropped at random.

        If `bank == "south"`, position is a uniform-random South cell
        (South's food is broadly distributed, so no clan-clustering is
        needed there). If `bank` is omitted entirely, position defaults
        to anywhere on the grid and the lifespan range is inferred from
        whichever bank the resolved `y` actually falls in.
        """
        agent_id = self._next_agent_id
        self._next_agent_id += 1

        if bank == "north":
            if x is None or y is None:
                open_hotspots = [
                    self.ecosystem.north_hotspots[i]
                    for i in range(self.ecosystem.north_hotspot_count)
                    if not self.ecosystem.is_hotspot_gorilla_occupied(i)
                ]
                spawn_pool = open_hotspots if open_hotspots else self.ecosystem.north_hotspots
                hotspot_x, hotspot_y = self._rng.choice(spawn_pool)
                radius = self.north_clan_spawn_radius
                if x is None:
                    x = hotspot_x + self._rng.randint(-radius, radius)
                if y is None:
                    y = hotspot_y + self._rng.randint(-radius, radius)
                x, y = self.ecosystem.clamp_to_bounds(x, y)
                # Never let the clan spawn spill south across the river.
                y = max(y, self.ecosystem.river_y + 1)
        elif bank == "south":
            if x is None:
                x = self._rng.randint(0, self.ecosystem.width - 1)
            if y is None:
                y = self._rng.randint(0, self.ecosystem.river_y)
        else:
            if x is None:
                x = self._rng.randint(0, self.ecosystem.width - 1)
            if y is None:
                y = self._rng.randint(0, self.ecosystem.height - 1)

        resolved_bank = bank if bank is not None else self.ecosystem.get_bank(y)
        max_age = self._draw_max_age_for_bank(resolved_bank)
        g_size, g_fertility = self._draw_genes_for_bank(resolved_bank)

        agent = EvolvableAgent(
            agent_id=agent_id,
            x=x,
            y=y,
            forced_g_t=forced_g_t,
            initial_energy=initial_energy,
            generation=generation,
            max_age=max_age,
            g_size=g_size,
            g_fertility=g_fertility,
            rng=self._rng,
        )
        self.agents_by_id[agent_id] = agent
        name = self._agent_name(agent_id)
        if name not in self.possible_agents:
            self.possible_agents.append(name)
        return agent

    def _draw_max_age_for_bank(self, bank: str) -> Optional[int]:
        """Randomly draw a lifespan cap for a newly-created agent by bank.

        Returns None (no lifespan cap) if the relevant range is None.
        """
        age_range = self.north_max_age_range if bank == "north" else self.south_max_age_range
        if age_range is None:
            return None
        low, high = age_range
        return self._rng.randint(low, high)

    def _draw_genes_for_bank(self, bank: str) -> Tuple[float, float]:
        """Draw (g_size, g_fertility) for a newly-spawned agent by bank.

        North (chimpanzee): larger-bodied (higher metabolic cost, stronger
        in combat) and high-fertility (fast, alpha-driven breeding).
        South (bonobo): lighter-bodied (cheaper metabolism) and
        low-fertility (slow, selective, socially-gated breeding). A small
        random jitter around each mean seeds genetic diversity so
        selection and mutation have variation to act on.
        """
        if bank == "north":
            size_mean, fert_mean = self.north_gene_means
        else:
            size_mean, fert_mean = self.south_gene_means
        g_size = max(0.5, min(1.6, size_mean + self._rng.gauss(0.0, 0.05)))
        g_fertility = max(0.0, min(1.0, fert_mean + self._rng.gauss(0.0, 0.05)))
        return g_size, g_fertility

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

        # Demographic split: explicit north/south counts take precedence
        # (precise calibration, e.g. "80 North, 60 South") when both are
        # given; otherwise fall back to the ratio-of-total split (North
        # gets the larger default cohort to absorb its higher expected
        # mortality, South the smaller, more stable cohort).
        if self.initial_north_population is not None and self.initial_south_population is not None:
            north_count = self.initial_north_population
            south_count = self.initial_south_population
        else:
            north_count = round(self._initial_population * self.north_population_ratio)
            south_count = self._initial_population - north_count

        for _ in range(north_count):
            self._spawn_agent(bank="north")
        for _ in range(south_count):
            self._spawn_agent(bank="south")

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
                brain_action = agent.decide_action(
                    food_present=food_present, rival_present=rival_present
                )
                chosen[aid] = self._apply_behavioral_override(agent, brain_action, food_present)

        # --- 2. Metabolism phase ---
        for aid in living_ids:
            agent = self.agents_by_id[aid]
            if agent.is_alive:
                agent.apply_metabolism(chosen[aid])

        # --- 2b. Aging phase ---
        # Every agent still alive after paying this step's metabolic cost
        # ages by one tick; agents that reach their lifespan cap die of
        # old age here, independent of energy — a North-bank Chimp with
        # full energy can still be retired for high combat/stress
        # wear-and-tear, while a long-lived South-bank Bonobo simply has
        # a much larger cap (or effectively none, if configured that way).
        for aid in living_ids:
            agent = self.agents_by_id[aid]
            if not agent.is_alive:
                continue
            agent.increment_age()
            # Count down the inter-birth interval each tick.
            agent.tick_reproduction_cooldown()
            if agent.is_expired():
                agent.is_alive = False
                log.append(
                    f"AGING: Agent {agent.agent_id} died of old age "
                    f"(age={agent.age}, max_age={agent.max_age})."
                )

        # --- 2c. Crowding-stress phase (North only) ---
        # Density-dependent regulation: North chimps packed too tightly
        # around a single patch accrue soft stress proportional to how far
        # local crowd exceeds `crowding_soft_cap`. This raises their stress
        # (feeding the fuzzy brain toward leaving/aggression) WITHOUT
        # directly killing them — the "get out of the mob" pressure. South
        # bonobos are exempt: their cohesive grouping is the whole point of
        # that bank. The density map is also cached here for reuse by the
        # migration logic in the individual-resolution phase below.
        self._density_map = self._build_density_map()
        self._local_crowd_cache = {}
        for aid in living_ids:
            agent = self.agents_by_id[aid]
            if not agent.is_alive:
                continue
            if self.ecosystem.get_bank(agent.y) != "north":
                continue
            crowd = self._local_crowd(agent, self._density_map)
            self._local_crowd_cache[aid] = crowd
            excess = crowd - self.crowding_soft_cap
            if excess > 0:
                added = excess * self.crowding_stress_per_excess
                agent.stress = max(0.0, min(100.0, agent.stress + added))

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
        # Per-bank living counts, computed once, so a fertile agent in a
        # crashed bank can be routed toward a mate (mate-seeking) exactly
        # like an overcrowded agent is routed toward an emptier patch.
        north_pop = self._bank_population("north")
        south_pop = self._bank_population("south")
        low_pop_threshold = self.genetic_engine.low_population_threshold

        for aid in living_ids:
            agent = self.agents_by_id[aid]
            if not agent.is_alive or suppress.get(aid, False):
                continue

            # Movement target resolution, in priority order:
            #   (a) mate-seeking: if this agent's OWN bank has crashed to
            #       <= low_pop_threshold and the agent is a ready breeder,
            #       steer it toward the nearest fertile partner so the
            #       widened low-population mating range yields real
            #       encounters instead of two survivors wandering a huge
            #       empty bank forever.
            #   (b) migration: else, if severely overcrowded (North), steer
            #       toward the nearest reachable open hotspot.
            move_target = None
            agent_bank = self.ecosystem.get_bank(agent.y)
            bank_pop = north_pop if agent_bank == "north" else south_pop
            is_ready_breeder = (
                self.genetic_engine.is_fertile(agent, self._reproduction_threshold_for(agent))
                and not agent.is_on_reproduction_cooldown()
            )
            if bank_pop <= low_pop_threshold and is_ready_breeder:
                move_target = self._nearest_fertile_partner(agent)

            if move_target is None:
                crowd = self._local_crowd_cache.get(aid)
                if crowd is not None and crowd >= self.crowding_migration_trigger:
                    move_target = self._nearest_open_hotspot(agent)

            action = chosen[aid]
            # A steering agent (mate-seeking OR migrating) commits to moving
            # toward its target. Its fuzzy brain may pick ATTACK/COOPERATE —
            # but if there's no agent actually on its cell to act on, those
            # are wasted no-ops that leave it drifting instead of closing
            # the distance. So when steering: redirect EAT/IDLE
            # unconditionally, and redirect ATTACK/COOPERATE too UNLESS
            # someone is genuinely co-located (a real fight/alliance we
            # shouldn't cancel).
            if move_target is not None:
                if action in (ActionType.EAT, ActionType.IDLE):
                    action = ActionType.MOVE
                elif action in (ActionType.ATTACK, ActionType.COOPERATE) and not self._has_colocated_agent(agent):
                    action = ActionType.MOVE

            self._apply_individual_action(agent, action, migration_target=move_target)

        # --- 4b. Gorilla-displacement phase ---
        # After movement resolves, any chimp now standing inside a
        # gorilla-occupied hotspot is displaced by the resident silverback
        # troop: it takes a stress hit plus a small energy cost (the
        # exertion/threat of being chased off) and is repelled to an
        # adjacent free cell. Gorillas never actually fight — their sheer
        # presence is an "invisible wall" the chimps simply cannot hold
        # ground against. This is what keeps the richest patches
        # permanently off-limits and fragments the chimp population.
        log.extend(self._apply_gorilla_displacement(living_ids))

        # --- 5. Ecosystem tick ---
        # Supply current North chimp positions so migrating gorilla troops
        # can avoid relocating straight onto a dense chimp colony.
        chimp_positions = [
            (a.x, a.y)
            for a in self.agents_by_id.values()
            if a.is_alive and self.ecosystem.get_bank(a.y) == "north"
        ]
        self.ecosystem.step(chimp_positions=chimp_positions)

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
    def _apply_gorilla_displacement(self, living_ids: List[int]) -> List[str]:
        """Displace any chimp standing inside a gorilla-occupied hotspot.

        Each affected agent takes `gorilla_stress_penalty` stress and
        `gorilla_energy_penalty` energy, then is repelled to an adjacent
        legal cell (reusing the same repel routine combat losers use).
        Returns a (usually short) log of displacement events.
        """
        log: List[str] = []
        for aid in living_ids:
            agent = self.agents_by_id.get(aid)
            if agent is None or not agent.is_alive:
                continue
            if self.ecosystem.get_bank(agent.y) != "north":
                continue
            if self.ecosystem.gorilla_hotspot_index_at(agent.x, agent.y) is None:
                continue

            agent.stress = max(0.0, min(100.0, agent.stress + self.gorilla_stress_penalty))
            agent.energy = max(0.0, min(100.0, agent.energy - self.gorilla_energy_penalty))
            agent.hunger = 100.0 - agent.energy
            if agent.energy <= 0.0:
                agent.is_alive = False
                log.append(f"GORILLA: Agent {agent.agent_id} did not survive being driven off a gorilla patch.")
                continue

            self._repel_from_gorilla(agent)

        return log

    def _repel_from_gorilla(self, agent: EvolvableAgent) -> None:
        """Push a displaced chimp to an adjacent legal cell OUTSIDE any gorilla zone.

        Prefers a neighboring cell that is both on the same bank (never
        crossing the river) and not itself gorilla-occupied, so the chimp
        is genuinely pushed out rather than shuffled within the forbidden
        patch. Falls back to any legal neighbor if fully surrounded.
        """
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        self._rng.shuffle(offsets)

        fallback: Optional[Tuple[int, int]] = None
        for dx, dy in offsets:
            target_x, target_y = agent.x + dx, agent.y + dy
            if not self.ecosystem.is_within_bounds(target_x, target_y):
                continue
            if not self.ecosystem.is_move_legal(agent.y, target_y):
                continue
            if fallback is None:
                fallback = (target_x, target_y)
            if self.ecosystem.gorilla_hotspot_index_at(target_x, target_y) is None:
                agent.x, agent.y = target_x, target_y
                return

        if fallback is not None:
            agent.x, agent.y = fallback

    def _apply_individual_action(
        self,
        agent: EvolvableAgent,
        action: ActionType,
        migration_target: Optional[Tuple[int, int]] = None,
    ) -> None:
        if action == ActionType.MOVE:
            self._move_agent(agent, migration_target=migration_target)
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

    def _is_in_north_hotspot(self, agent: EvolvableAgent) -> bool:
        """True if the agent is on the North bank AND inside a food hotspot.

        Used to grant the reproduction discount described in
        GeneticEngine (`north_hotspot_fertility_threshold` /
        `north_hotspot_reproduction_cost`): Chimpanzees holding a
        productive patch breed faster to counteract higher mortality.
        """
        if self.ecosystem.get_bank(agent.y) != "north":
            return False
        return self.ecosystem.is_in_north_hotspot(agent.x, agent.y)

    def _has_colocated_agent(self, agent: EvolvableAgent) -> bool:
        """True if any OTHER living agent shares this agent's exact cell.

        This is the actual mechanical requirement for ATTACK/COOPERATE to
        have any effect at all: `InteractionResolver.resolve_cell` only
        resolves interactions for agents sharing the exact same (x, y).
        Sensing a rival within `perception_radius` is NOT the same thing
        — a rival can be "present" for fuzzy-brain purposes while being
        two or three cells away, in which case ATTACK/COOPERATE is a
        guaranteed no-op this step.
        """
        for other in self.agents_by_id.values():
            if other.agent_id == agent.agent_id or not other.is_alive:
                continue
            if other.x == agent.x and other.y == agent.y:
                return True
        return False

    def _apply_behavioral_override(
        self, agent: EvolvableAgent, action: ActionType, food_present: bool
    ) -> ActionType:
        """Correct two confirmed no-op traps in the Phase 3 fuzzy brain.

        CONFIRMED BUG (root-caused via direct step-by-step tracing):
        `rival_present`, which the fuzzy brain uses to weight ATTACK, is
        computed from `perception_radius` — a rival can be "sensed" while
        still 2-3 cells away. But ATTACK/COOPERATE only have a mechanical
        effect when an agent shares the EXACT same cell as another agent
        (see `InteractionResolver.resolve_cell`). A hungry, aggressive
        agent can therefore repeatedly choose ATTACK against a rival it
        can sense but never actually reach, paying the full -10 energy
        cost each time with zero effect — freezing in place (ATTACK
        performs no movement) and starving to death within a handful of
        steps. Traced example: agent stayed at a fixed cell for 4
        consecutive steps, chose ATTACK every time with no one actually
        co-located, and died from pure self-inflicted metabolic cost
        (41 -> 31 -> 21 -> 11 -> 1 -> dead), with no combat ever
        occurring. Rule: if ATTACK/COOPERATE is chosen but no one is
        actually standing on this agent's cell, it is unconditionally
        redirected to MOVE — there is no scenario where repeating the
        no-op action is better than closing the distance.

        SEPARATE FORAGING GAP (from the earlier extinction-crisis fix):
        if the agent is hungry, has no food on its own tile, but CAN see
        food somewhere within `foraging_radius`, prioritize walking
        toward it over an EAT/COOPERATE/ATTACK attempt that accomplishes
        nothing locally.
        """
        if action in (ActionType.ATTACK, ActionType.COOPERATE) and not self._has_colocated_agent(
            agent
        ):
            return ActionType.MOVE

        if food_present:
            return action
        if action not in (ActionType.EAT, ActionType.COOPERATE, ActionType.ATTACK):
            return action
        if agent.hunger < self.foraging_override_hunger_threshold:
            return action
        if self._find_nearest_food(agent) is None:
            return action
        return ActionType.MOVE

    def _build_density_map(self) -> Dict[Tuple[int, int], int]:
        """Return a map of cell -> living-agent count, computed once per step.

        Used to answer "how crowded is this agent's neighborhood" cheaply:
        instead of an O(n) scan per agent (O(n^2) overall), we bucket all
        agents into their cells once, then each agent sums the buckets in
        its `crowding_radius` neighborhood.
        """
        density: Dict[Tuple[int, int], int] = {}
        for agent in self.agents_by_id.values():
            if not agent.is_alive:
                continue
            key = (agent.x, agent.y)
            density[key] = density.get(key, 0) + 1
        return density

    def _local_crowd(self, agent: EvolvableAgent, density_map: Dict[Tuple[int, int], int]) -> int:
        """Count living agents within `crowding_radius` of this agent (incl. self)."""
        r = self.crowding_radius
        total = 0
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                total += density_map.get((agent.x + dx, agent.y + dy), 0)
        return total

    def _bank_population(self, bank: str) -> int:
        """Count living agents currently on the given bank."""
        return sum(
            1
            for a in self.agents_by_id.values()
            if a.is_alive and self.ecosystem.get_bank(a.y) == bank
        )

    def _nearest_fertile_partner(self, agent: EvolvableAgent) -> Optional[Tuple[int, int]]:
        """Return the coordinate of the nearest eligible mating partner for
        `agent`, searched within `low_population_mating_distance` and
        restricted to the SAME bank (the river is an impassable divide).

        A valid partner must be a different living agent, energy-fertile,
        not on reproduction cooldown, and not currently overlapping this
        agent's own cell (if they already share a cell, they don't need to
        move toward each other). This is the "mate-seeking compass": when a
        bank's population has crashed, its fertile survivors actively steer
        toward one another instead of wandering blindly, which is what
        actually lets the wide low-population mating distance translate into
        real encounters and births.
        """
        my_bank = self.ecosystem.get_bank(agent.y)
        search_radius = self.genetic_engine.low_population_mating_distance

        best_coord: Optional[Tuple[int, int]] = None
        best_distance: Optional[int] = None
        for other in self.agents_by_id.values():
            if other.agent_id == agent.agent_id or not other.is_alive:
                continue
            if self.ecosystem.get_bank(other.y) != my_bank:
                continue
            if not self.genetic_engine.is_fertile(other, self._reproduction_threshold_for(other)):
                continue
            if other.is_on_reproduction_cooldown():
                continue
            distance = max(abs(other.x - agent.x), abs(other.y - agent.y))
            if distance == 0 or distance > search_radius:
                continue
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_coord = (other.x, other.y)
        return best_coord

    def _nearest_open_hotspot(self, agent: EvolvableAgent) -> Optional[Tuple[int, int]]:
        """Return the coordinate of the nearest non-gorilla, non-depleted
        North hotspot within `migration_vision_radius`, excluding the one
        the agent is essentially already standing on.

        This is the "migration compass": when a chimp is too crowded, it
        can see far past its normal short foraging range to locate a less
        contested patch to head toward, so density pressure actually
        results in dispersal to real food rather than aimless wandering
        into empty desert.
        """
        best_coord: Optional[Tuple[int, int]] = None
        best_distance: Optional[int] = None
        for index, (hx, hy) in enumerate(self.ecosystem.north_hotspots):
            state = self.ecosystem.hotspot_state[index]
            if state["gorilla_occupied"] or state["depleted"]:
                continue
            distance = max(abs(hx - agent.x), abs(hy - agent.y))
            if distance > self.migration_vision_radius:
                continue
            # Skip the patch it's already sitting on (radius+1 slack), so
            # "migrate" always means "go somewhere else".
            if distance <= self.ecosystem.north_hotspot_radius:
                continue
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_coord = (hx, hy)
        return best_coord

    def _find_nearest_food(self, agent: EvolvableAgent):
        """Return the nearest visible FoodItem within `foraging_radius`, or None.

        Distance is measured via Chebyshev distance (matching the
        8-connected movement grid), so "nearest" corresponds directly to
        "fewest MOVE steps away" rather than straight-line distance.
        """
        nearest_item = None
        nearest_distance = None

        for item in self.ecosystem.food_items:
            distance = max(abs(item.x - agent.x), abs(item.y - agent.y))
            if distance > self.foraging_radius:
                continue
            if nearest_item is None or distance < nearest_distance:
                nearest_item = item
                nearest_distance = distance

        return nearest_item

    @staticmethod
    def _sign(value: int) -> int:
        """Return -1, 0, or 1 for the sign of an integer delta."""
        return (value > 0) - (value < 0)

    def _move_agent(self, agent: EvolvableAgent, migration_target: Optional[Tuple[int, int]] = None) -> None:
        """Move an agent one cell.

        Priority:
          1. If a `migration_target` is supplied (the agent is overcrowded
             and there's a reachable less-contested open hotspot), steer
             toward that target. This is the "escape the mob, head to the
             emptier patch" behavior — it overrides local food-seeking so a
             crowded agent actually leaves instead of orbiting the same
             exhausted cell.
          2. Otherwise, if food is visible within `foraging_radius`, steer
             toward the nearest item with probability `food_seeking_bias`.
          3. Otherwise, random legal walk (exploration).
        """
        all_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        candidate_order: List[Tuple[int, int]]

        if migration_target is not None:
            tx, ty = migration_target
            preferred = (self._sign(tx - agent.x), self._sign(ty - agent.y))
            if preferred != (0, 0):
                remaining = [o for o in all_offsets if o != preferred]
                self._rng.shuffle(remaining)
                candidate_order = [preferred] + remaining
            else:
                candidate_order = list(all_offsets)
                self._rng.shuffle(candidate_order)
        else:
            target_food = self._find_nearest_food(agent)
            if target_food is not None:
                preferred = (self._sign(target_food.x - agent.x), self._sign(target_food.y - agent.y))
                if preferred != (0, 0) and self._rng.random() < self.food_seeking_bias:
                    remaining = [o for o in all_offsets if o != preferred]
                    self._rng.shuffle(remaining)
                    candidate_order = [preferred] + remaining
                else:
                    candidate_order = list(all_offsets)
                    self._rng.shuffle(candidate_order)
            else:
                candidate_order = list(all_offsets)
                self._rng.shuffle(candidate_order)

        for dx, dy in candidate_order:
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
    def _reproduction_threshold_for(self, agent: EvolvableAgent) -> Optional[float]:
        """Return the hotspot-discounted fertility threshold, or None for default."""
        if self._is_in_north_hotspot(agent):
            return self.genetic_engine.north_hotspot_fertility_threshold
        return None

    def _reproduction_cost_for(self, agent: EvolvableAgent) -> Optional[float]:
        """Return the hotspot-discounted reproduction cost, or None for default."""
        if self._is_in_north_hotspot(agent):
            return self.genetic_engine.north_hotspot_reproduction_cost
        return None

    def _reproduction_cooldown_for(self, agent: EvolvableAgent) -> int:
        """Return the post-birth cooldown (inter-birth interval) for `agent`'s bank.

        North (chimp) recovers fast (short cooldown) to replace its high
        losses; South (bonobo) stays damped (long cooldown) for stability.
        """
        if self.ecosystem.get_bank(agent.y) == "north":
            return self.north_reproduction_cooldown
        return self.south_reproduction_cooldown

    def _compute_child_position(self, parent: EvolvableAgent) -> Tuple[int, int]:
        """Decide where a newborn spawns relative to its parent.

        NORTH (Chimp): offspring disperse — placed at a random offset up
        to `north_birth_dispersal_radius` cells from the parent (kept on
        the North bank, in bounds, and OUT of gorilla zones where
        possible). This models real chimpanzee fission: juveniles and
        females scatter to find their own patches rather than piling onto
        the parent's exact cell. It's the direct fix for the observed
        "rich-get-richer" super-colony, where every birth stacking on one
        coordinate let a single hotspot swell to thousands of agents.

        SOUTH (Bonobo): offspring stay put (born on the parent's cell),
        modeling cohesive bonobo groups that don't fragment — which is
        exactly what lets them form the stable, bonded society the South
        bank is meant to represent.
        """
        if self.ecosystem.get_bank(parent.y) != "north":
            return (parent.x, parent.y)

        radius = self.north_birth_dispersal_radius
        best: Optional[Tuple[int, int]] = None
        for _ in range(8):  # a few attempts to land a non-gorilla, legal cell
            dx = self._rng.randint(-radius, radius)
            dy = self._rng.randint(-radius, radius)
            tx, ty = parent.x + dx, parent.y + dy
            tx, ty = self.ecosystem.clamp_to_bounds(tx, ty)
            # Keep the child on the North bank (never across the river).
            if self.ecosystem.get_bank(ty) != "north":
                ty = max(ty, self.ecosystem.river_y + 1)
            if best is None:
                best = (tx, ty)
            if self.ecosystem.gorilla_hotspot_index_at(tx, ty) is None:
                return (tx, ty)

        return best if best is not None else (parent.x, parent.y)

    def _process_reproduction(self, chosen_actions: Dict[int, ActionType]) -> List[str]:
        log: List[str] = []

        living = [a for a in self.agents_by_id.values() if a.is_alive]
        # Per-bank counts: a bank's own crash must trigger its own rescue
        # rules, even if the OTHER bank is healthy and the total looks fine.
        # (This was the South-extinction bug: South collapsed to 2 while a
        # healthy North kept the total above the threshold, so neither the
        # mating-distance boost nor the conception guarantee ever engaged.)
        north_pop = sum(1 for a in living if self.ecosystem.get_bank(a.y) == "north")
        south_pop = len(living) - north_pop

        def bank_pop_for(agent: EvolvableAgent) -> int:
            return north_pop if self.ecosystem.get_bank(agent.y) == "north" else south_pop

        fertile_sorted = sorted(
            (
                a
                for a in living
                if self.genetic_engine.is_fertile(a, self._reproduction_threshold_for(a))
            ),
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

                # Use the SMALLER of the two parents' bank populations for
                # the low-population rules (in practice both parents are on
                # the same bank, since the river blocks cross-bank mating).
                pair_bank_pop = min(bank_pop_for(parent1), bank_pop_for(parent2))

                can_mate = self.genetic_engine.can_mate(
                    parent1,
                    parent2,
                    action1,
                    action2,
                    threshold1=self._reproduction_threshold_for(parent1),
                    threshold2=self._reproduction_threshold_for(parent2),
                    current_population=pair_bank_pop,
                )
                if not can_mate:
                    continue

                # Fertility-strategy gate: the pair is eligible, but whether
                # conception actually happens is rolled against the parents'
                # g_fertility genes — UNLESS that parent's own bank has
                # crashed to <= low_population_threshold, in which case
                # conception is guaranteed so a near-extinct bank isn't also
                # fighting the fertility dice. A failed roll marks both as
                # "used" this step (they attempted) but produces no child.
                low_pop = pair_bank_pop <= self.genetic_engine.low_population_threshold
                if not low_pop and not self.genetic_engine.roll_conception(parent1, parent2):
                    used_ids.add(parent1.agent_id)
                    used_ids.add(parent2.agent_id)
                    break

                child_position = self._compute_child_position(parent1)
                child = self.genetic_engine.reproduce(
                    parent1, parent2, self._next_agent_id, child_position
                )
                self._next_agent_id += 1

                # Lifespan policy is an arena/ecology concern, not a
                # genetics concern, so it's assigned here based on the
                # bank the child is actually born into (not inherited
                # from either parent).
                child_bank = self.ecosystem.get_bank(child.y)
                child.max_age = self._draw_max_age_for_bank(child_bank)

                self.agents_by_id[child.agent_id] = child
                self.possible_agents.append(self._agent_name(child.agent_id))

                self.genetic_engine.apply_reproduction_cost(
                    parent1, self._reproduction_cost_for(parent1)
                )
                self.genetic_engine.apply_reproduction_cost(
                    parent2, self._reproduction_cost_for(parent2)
                )

                # Inter-birth interval: each parent enters a cooldown sized
                # to ITS OWN bank — North chimps recover fast (short), South
                # bonobos stay damped (long) — so neither can conceive again
                # until its interval elapses.
                p1_cd = self._reproduction_cooldown_for(parent1)
                p2_cd = self._reproduction_cooldown_for(parent2)
                if p1_cd > 0:
                    parent1.reproduction_cooldown = p1_cd
                if p2_cd > 0:
                    parent2.reproduction_cooldown = p2_cd

                used_ids.add(parent1.agent_id)
                used_ids.add(parent2.agent_id)

                log.append(
                    f"REPRODUCTION: Agent {parent1.agent_id} x Agent {parent2.agent_id} "
                    f"-> offspring Agent {child.agent_id} "
                    f"(generation {child.generation}, bank={child_bank}, "
                    f"max_age={child.max_age}, G_T={child.g_t:.3f}, G_E={child.g_e:.3f}, "
                    f"size={child.g_size:.2f}, fert={child.g_fertility:.2f})"
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