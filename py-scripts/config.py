"""
scripts/config.py

Centralized configuration for the Congo River Paradigm simulation.

ALL tunable scientific parameters live here, grouped by subsystem, so an
experiment can be defined in one place instead of hunting through the code.
`run_simulation.build_arena()` reads exclusively from this module, which
makes parameter sweeps and reproducibility studies straightforward: copy
this file, change a few numbers, and you have a new named experiment.

Structure:
    ECOSYSTEM_CONFIG  -> world: grid, food spawning, gorillas, decay
    GENETICS_CONFIG   -> reproduction, crossover, mutation, fertility
    ARENA_CONFIG      -> population, lifespans, crowding, migration, genes
    SIM_CONFIG        -> top-level run settings (seed, perception)

Each value carries an inline comment explaining WHAT it controls and WHY
it's set the way it is, so the file doubles as living documentation.
"""

from __future__ import annotations

from typing import Optional, Tuple

# ======================================================================
# ⭐ EXPERIMENT KNOBS — THE PARAMETERS WE TUNE MOST OFTEN ⭐
# ======================================================================
# These are the "dials" for our scientific experiments (parameter sweeps).
# They're pulled to the very top so we never have to scroll hunting for
# them. Each one is wired down into the real config dicts below, so
# changing a value HERE changes the whole simulation — one edit, one place.
#
# For a sweep: change ONE of these, keep the rest fixed, run many seeds,
# and compare outcomes. Only change one knob at a time so any effect you
# see can be attributed to that single variable.
#
#   FOOD_SCARCITY  → the core hypothesis: does less food breed aggression?
#   GORILLA_COUNT  → competition pressure: how many patches are blocked?
#   SOUTH_START    → Founder Effect: does a small start doom the bonobos?
#   SOUTH_FOOD     → the "paradise makes you lazy?" test (your new idea)
#   MUTATION       → evolutionary speed: faster drift with more mutation?
# ----------------------------------------------------------------------

# 🍎 North food richness — LOWER = scarcer = (hypothesis) more aggression.
#    Sweep suggestion: [15, 25, 35, 50, 70]   (baseline = 35)
FOOD_SCARCITY = 35.0

# 🦍 How many of the 4 North hotspots gorillas permanently block.
#    Sweep suggestion: [0, 1, 2, 3]           (baseline = 2)
GORILLA_COUNT = 2

# 👥 Starting bonobo population (Founder Effect / extinction risk).
#    Sweep suggestion: [10, 20, 60, 100]      (baseline = 60)
SOUTH_START = 60

# 🌴 South food richness — for the "abundance suppresses breeding?" test.
#    Sweep suggestion: [25, 40, 60, 80]       (baseline = 40)
SOUTH_FOOD = 40.0

# 🧬 Mutation rate — how fast genes drift each generation.
#    Sweep suggestion: [0.01, 0.05, 0.10]     (baseline = 0.05)
MUTATION = 0.05

# ======================================================================


# ======================================================================
# ECOSYSTEM — the physical world: grid, food, gorillas, decay
# ======================================================================
ECOSYSTEM_CONFIG = {
    # --- Grid & river ---
    "width": 50,
    "height": 50,
    "river_y": 25,                       # impassable divide: North (Y>25) vs South (Y<=25)

    # --- Food spawning (the core North-scarce / South-abundant asymmetry) ---
    "north_spawn_prob": 0.55,            # raised: only 2 of 4 patches are open (gorillas hold 2),
                                         # so open patches must produce more to feed the population
    "south_spawn_prob": 0.50,
    "north_food_energy": FOOD_SCARCITY,   # ⭐ wired from EXPERIMENT KNOBS (food scarcity test)
    "south_food_energy": SOUTH_FOOD,      # ⭐ wired from EXPERIMENT KNOBS (abundance-vs-breeding test)
    "south_cluster_size_range": (2, 4),
    "south_cluster_radius": 2,
    "south_isolated_fruit_chance": 0.30,  # South stays broadly uniform/stable (no gorillas)

    # --- North hotspots (choke-points that force competition) ---
    "north_hotspot_count": 4,            # 2 gorilla-held + 2 open/contested
    "north_hotspot_radius": 3,
    "north_hotspot_spawn_size": (4, 8),  # dense per-firing drop, offsetting the halved open-patch count

    # --- Patch depletion cycle (open patches exhaust and recover) ---
    "depletion_threshold": 800.0,        # open patches are the ONLY food, so exhaust them slowly
    "depletion_recovery_steps": 20,      # short dark period so a depleted patch returns quickly

    # --- Gorilla system ---
    "gorilla_occupied_count": GORILLA_COUNT,  # ⭐ wired from EXPERIMENT KNOBS (competition pressure test)
    "gorilla_forage_rate": 3.0,          # how fast a resident troop eats down its patch
    "gorilla_migration_threshold": 1200.0,  # foraged-energy a troop consumes before moving on
    "gorilla_min_residence_steps": 400,  # min steps a troop stays put -> chimps get time to settle

    # --- Midway "travel snack" food (keeps migrating chimps alive between patches) ---
    "midway_food_prob": 0.15,            # sparse: travel snacks appear on only ~15% of steps
    "midway_food_energy": 4.0,           # low-value filler (~1/9 of a real hotspot meal)
    "midway_food_max_per_step": 2,       # at most 2 snacks/step -> corridor stays lean, not a patch

    # --- Food decay / rot ---
    "food_decay_steps": 100,             # uneaten food rots after ~100 steps (fresh->aging->rotting)
}

# ======================================================================
# GENETICS — reproduction, crossover, mutation, fertility
# ======================================================================
GENETICS_CONFIG = {
    "reproduction_energy_threshold": 65.0,  # energy an agent needs to be eligible to breed
    "reproduction_cost": 22.0,              # energy spent per birth (default, non-hotspot)
    "crossover_weight_parent1": 0.6,        # parent-1 gene dominance in the blend (0.6 / 0.4)
    "mutation_rate": MUTATION,              # ⭐ wired from EXPERIMENT KNOBS (evolutionary speed test)
    "mutation_sigma": 0.05,                 # magnitude (std-dev) of a Gaussian mutation nudge
    "offspring_initial_energy": 50.0,       # newborns start at half energy
    "mating_max_distance": 1,               # normal mating range (adjacent cells)
    "north_hotspot_fertility_threshold": 45.0,  # North breeds while holding an open patch (discounted)
    "north_hotspot_reproduction_cost": 18.0,
    "low_population_threshold": 5,          # bank pop <= this -> "crisis" mate-seeking + guaranteed conception
    "low_population_mating_distance": 15,   # dramatically widened search range in a crisis
    "reproduction_cooldown_steps": 60,      # default inter-birth interval (arena overrides per-bank below)
}

# ======================================================================
# ARENA — population, lifespans, crowding, migration, per-bank genes
# ======================================================================
ARENA_CONFIG = {
    # --- Starting populations (Founder Effect: big North, small South) ---
    "initial_north_population": 30,      # sized to grow into the 2 open patches, not shock-collapse
    "initial_south_population": SOUTH_START,  # ⭐ wired from EXPERIMENT KNOBS (Founder Effect test)

    # --- Lifespans (bank-specific aging) ---
    # North chimps are stress/combat-worn but live long enough to reproduce first.
    "north_max_age_range": (200, 350),
    # South bonobos now AGE too (realistic — real bonobos are not immortal), but with a
    # LONG lifespan: bonobos are slow, K-selected breeders, so they need many steps to
    # reproduce before dying. Validated at 4/5 seed survival with mate-seeking active.
    # (Set to None to disable South aging entirely, as in earlier builds.)
    "south_max_age_range": (600, 1000),

    # --- Spawn / dispersal ---
    "north_clan_spawn_radius": 5,        # troupe starts living AROUND its open hotspot
    "north_birth_dispersal_radius": 6,   # newborn Chimps scatter (fission), breaking super-colonies

    # --- Gorilla displacement penalties (applied to chimps in a gorilla zone) ---
    "gorilla_stress_penalty": 15.0,      # meaningful stress hit, not an instant death sentence
    "gorilla_energy_penalty": 1.0,       # light energy cost of being chased off

    # --- Crowding / density regulation (North) ---
    "crowding_radius": 3,                # neighborhood size for measuring local density
    "crowding_soft_cap": 8,              # crowd beyond this starts adding soft stress (no direct death)
    "crowding_stress_per_excess": 1.5,   # stress added per agent over the soft cap
    "crowding_migration_trigger": 10,    # local crowd at/above this makes an agent seek another patch
    "migration_vision_radius": 45,       # must exceed inter-hotspot distance (~27) so the OTHER open
                                         # patch is always visible; grid is 50 wide, so 45 covers any pair

    # --- Per-bank independent genes (size, fertility) ---
    "north_gene_means": (1.2, 0.85),     # chimp: heavy body (costly metabolism, strong), high fertility
    "south_gene_means": (0.85, 0.30),    # bonobo: light body (cheap metabolism), low/selective fertility

    # --- Per-bank inter-birth interval ---
    "north_reproduction_cooldown": 20,   # SHORT: chimps breed fast to replace high losses (r-selected)
    "south_reproduction_cooldown": 60,   # LONG: bonobos stay damped toward equilibrium (K-selected)

    # --- Foraging behavior ---
    "foraging_radius": 5,
    "food_seeking_bias": 0.9,
}

# ======================================================================
# SIM — top-level run settings
# ======================================================================
SIM_CONFIG = {
    "rng_seed": 2026,                    # default seed; override per-run for sweeps / variance studies
    "perception_radius": 3,              # how far an agent senses its immediate surroundings
}