"""
scripts/experiment_runner.py

Batch experiment runner for the Congo River Paradigm simulation.

This is the scientific data-collection engine (Phase 5). It runs the
simulation many times in "headless" batch mode (NO Pygame window, so it's
fast), sweeping ONE parameter across several values while running many
random seeds per value, and writes two CSV files:

    <experiment>_timeseries.csv  -> per-step metrics (sampled every N steps)
                                    for gene-drift / population curves
    <experiment>_summary.csv     -> one row per run, the final outcome
                                    (survival, final genes, births, etc.)
                                    for parameter-vs-outcome comparisons

Design (one-factor-at-a-time, OFAT): to isolate cause and effect, each
experiment sweeps a SINGLE knob while every other parameter stays at its
config.py baseline. If an outcome shifts, it can be attributed to that one
variable.

Usage:
    python scripts/experiment_runner.py <experiment>

    where <experiment> is one of: food, gorilla, south_start, south_food,
    mutation  (see EXPERIMENTS below). For example:

        python scripts/experiment_runner.py food

    runs 5 food-scarcity values x 20 seeds = 100 runs and writes
    results/food_timeseries.csv and results/food_summary.csv.
"""

from __future__ import annotations

import csv
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

# --- Make the project root importable regardless of working directory ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.environment.arena import CongoArena  # noqa: E402
import config as config_module  # noqa: E402
from run_simulation import build_arena  # noqa: E402


# ======================================================================
# EXPERIMENT DESIGN — tune these to change what gets swept
# ======================================================================

# How long each individual run lasts, and how often we sample the
# timeseries (every SAMPLE_EVERY steps) to keep the timeseries file small
# while still capturing the shape of the gene-drift / population curves.
STEPS_PER_RUN = 3000
SAMPLE_EVERY = 10

# Fixed, reproducible seeds: rerunning an experiment reproduces it exactly.
SEEDS = list(range(1, 21))  # 20 seeds: 1, 2, ..., 20

# Where result CSVs are written.
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


@dataclass(frozen=True)
class Experiment:
    """Definition of a single one-factor-at-a-time parameter sweep.

    Attributes:
        name:         CLI name and output-file prefix (e.g. "food").
        config_dict:  Which config.py dict holds the knob being swept
                      (e.g. config.ECOSYSTEM_CONFIG).
        param_key:    The key inside that dict to overwrite each run
                      (e.g. "north_food_energy").
        values:       The list of values to sweep across.
        description:  Human-readable summary of the hypothesis, printed
                      when the experiment starts.
    """

    name: str
    config_dict_name: str
    param_key: str
    values: List[float]
    description: str


# The experiment registry. Each entry sweeps exactly ONE knob. The values
# mirror the sweep suggestions documented at the top of config.py.
EXPERIMENTS: Dict[str, Experiment] = {
    "food": Experiment(
        name="food",
        config_dict_name="ECOSYSTEM_CONFIG",
        param_key="north_food_energy",
        values=[15.0, 25.0, 35.0, 50.0, 70.0],
        description="Food scarcity vs aggression (the core Congo River hypothesis): "
        "does scarcer North food drive higher aggression?",
    ),
    "gorilla": Experiment(
        name="gorilla",
        config_dict_name="ECOSYSTEM_CONFIG",
        param_key="gorilla_occupied_count",
        values=[0, 1, 2, 3],
        description="Competition pressure: how does the number of gorilla-blocked "
        "hotspots affect North aggression and survival?",
    ),
    "south_start": Experiment(
        name="south_start",
        config_dict_name="ARENA_CONFIG",
        param_key="initial_south_population",
        values=[10, 20, 60, 100],
        description="Founder Effect: does a small starting bonobo population raise "
        "extinction risk?",
    ),
    "south_food": Experiment(
        name="south_food",
        config_dict_name="ECOSYSTEM_CONFIG",
        param_key="south_food_energy",
        values=[25.0, 40.0, 60.0, 80.0],
        description="Abundance vs breeding: does richer South food change the "
        "bonobo population trajectory?",
    ),
    "mutation": Experiment(
        name="mutation",
        config_dict_name="GENETICS_CONFIG",
        param_key="mutation_rate",
        values=[0.01, 0.05, 0.10],
        description="Evolutionary speed: does a higher mutation rate accelerate "
        "genetic divergence between the banks?",
    ),
}


# ======================================================================
# METRICS — how we measure a running / finished simulation
# ======================================================================

@dataclass
class RunAccumulator:
    """Collects metrics across a single run.

    Holds the sampled timeseries rows plus running totals (births/deaths)
    and the first-extinction steps for each bank, which are finalized into
    a summary row when the run ends.
    """

    timeseries_rows: List[dict] = field(default_factory=list)
    total_births: int = 0
    total_deaths: int = 0
    north_extinction_step: Optional[int] = None
    south_extinction_step: Optional[int] = None


def _bank_lists(arena: CongoArena):
    """Return (north_agents, south_agents) lists of the living agents."""
    north, south = [], []
    for agent in arena.agents_by_id.values():
        if not agent.is_alive:
            continue
        if arena.ecosystem.get_bank(agent.y) == "north":
            north.append(agent)
        else:
            south.append(agent)
    return north, south


def _mean(values: List[float]) -> float:
    """Mean that safely returns 0.0 for an empty list (extinct bank)."""
    return statistics.mean(values) if values else 0.0


def _snapshot_metrics(arena: CongoArena) -> dict:
    """Compute the current-step metrics for both banks.

    Returns a dict of population counts and average gene / energy / stress
    values per bank. Used both for timeseries sampling and for the final
    summary row.
    """
    north, south = _bank_lists(arena)
    return {
        "north_pop": len(north),
        "south_pop": len(south),
        # Aggression gene (the headline evolutionary signal)
        "avg_gt_north": round(_mean([a.g_t for a in north]), 4),
        "avg_gt_south": round(_mean([a.g_t for a in south]), 4),
        # Body-size gene
        "avg_size_north": round(_mean([a.g_size for a in north]), 4),
        "avg_size_south": round(_mean([a.g_size for a in south]), 4),
        # Bio-energetic state
        "avg_energy_north": round(_mean([a.energy for a in north]), 2),
        "avg_energy_south": round(_mean([a.energy for a in south]), 2),
        "avg_stress_north": round(_mean([a.stress for a in north]), 2),
        "avg_stress_south": round(_mean([a.stress for a in south]), 2),
    }


def _count_log_events(step_log: List[str], accumulator: RunAccumulator) -> None:
    """Scan a step's event log to tally births and deaths for this run."""
    for line in step_log:
        if "REPRODUCTION" in line:
            accumulator.total_births += 1
        # Deaths surface in the log as combat/predation/starvation/aging
        # outcomes; counting the explicit death markers covers them.
        if "died" in line or "did not survive" in line or "preys on" in line:
            accumulator.total_deaths += 1


# ======================================================================
# RUNNING ONE SIMULATION
# ======================================================================

def run_single(
    experiment: Experiment,
    param_value: float,
    seed: int,
    run_id: int,
) -> tuple[List[dict], dict]:
    """Run ONE simulation for a given swept value + seed.

    Overrides the swept parameter in its config dict, builds a fresh arena,
    steps it for STEPS_PER_RUN, samples the timeseries every SAMPLE_EVERY
    steps, and finalizes a summary row.

    Returns (timeseries_rows, summary_row). The caller is responsible for
    restoring the config afterward (see run_experiment).

    Returns:
        timeseries_rows: list of per-sample dicts (tagged with run context).
        summary_row: single dict describing the run's final outcome.
    """
    # --- Override the single swept parameter for this run ---
    config_dict = getattr(config_module, experiment.config_dict_name)
    config_dict[experiment.param_key] = param_value

    arena = build_arena(rng_seed=seed)
    arena.reset(seed=seed)

    acc = RunAccumulator()

    for _ in range(STEPS_PER_RUN):
        # Stop early only if EVERYTHING is dead (global extinction) — there's
        # nothing left to simulate. A single-bank extinction keeps running so
        # we still observe the surviving lineage.
        if not any(a.is_alive for a in arena.agents_by_id.values()):
            break

        arena.step()
        _count_log_events(arena.last_step_log, acc)

        north, south = _bank_lists(arena)
        # Record the first step at which each bank hits zero.
        if not north and acc.north_extinction_step is None:
            acc.north_extinction_step = arena.current_step
        if not south and acc.south_extinction_step is None:
            acc.south_extinction_step = arena.current_step

        # Sample the timeseries at the configured cadence.
        if arena.current_step % SAMPLE_EVERY == 0:
            row = {
                "run_id": run_id,
                "experiment": experiment.name,
                "param_value": param_value,
                "seed": seed,
                "step": arena.current_step,
                **_snapshot_metrics(arena),
            }
            acc.timeseries_rows.append(row)

    # --- Finalize the summary row ---
    final = _snapshot_metrics(arena)
    north_survived = final["north_pop"] > 0
    south_survived = final["south_pop"] > 0
    summary_row = {
        "run_id": run_id,
        "experiment": experiment.name,
        "param_value": param_value,
        "seed": seed,
        "steps_completed": arena.current_step,
        "north_survived": int(north_survived),
        "south_survived": int(south_survived),
        "coexistence": int(north_survived and south_survived),
        "final_north_pop": final["north_pop"],
        "final_south_pop": final["south_pop"],
        "final_avg_gt_north": final["avg_gt_north"],
        "final_avg_gt_south": final["avg_gt_south"],
        "final_avg_size_north": final["avg_size_north"],
        "final_avg_size_south": final["avg_size_south"],
        "final_avg_energy_north": final["avg_energy_north"],
        "final_avg_energy_south": final["avg_energy_south"],
        "final_avg_stress_north": final["avg_stress_north"],
        "final_avg_stress_south": final["avg_stress_south"],
        "total_births": acc.total_births,
        "total_deaths": acc.total_deaths,
        "extinction_step_north": acc.north_extinction_step
        if acc.north_extinction_step is not None
        else "",
        "extinction_step_south": acc.south_extinction_step
        if acc.south_extinction_step is not None
        else "",
    }
    return acc.timeseries_rows, summary_row


# ======================================================================
# ORCHESTRATING A FULL EXPERIMENT (all values x all seeds)
# ======================================================================

def run_experiment(experiment: Experiment) -> None:
    """Run a full sweep: every value x every seed, writing both CSVs.

    Progress is printed per-run with the run's headline outcome so the
    scientist can watch results form live. The swept config parameter is
    always restored to its original baseline afterward, even on error, so
    running one experiment never leaves the config mutated for the next.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timeseries_path = os.path.join(RESULTS_DIR, f"{experiment.name}_timeseries.csv")
    summary_path = os.path.join(RESULTS_DIR, f"{experiment.name}_summary.csv")

    total_runs = len(experiment.values) * len(SEEDS)
    print("=" * 78)
    print(f" EXPERIMENT: {experiment.name}")
    print(f" {experiment.description}")
    print("-" * 78)
    print(f" Sweeping {experiment.config_dict_name}['{experiment.param_key}'] "
          f"over {experiment.values}")
    print(f" {len(SEEDS)} seeds x {len(experiment.values)} values = {total_runs} runs, "
          f"{STEPS_PER_RUN} steps each")
    print("=" * 78)

    # Remember the baseline so we can restore it no matter what.
    config_dict = getattr(config_module, experiment.config_dict_name)
    original_value = config_dict[experiment.param_key]

    timeseries_rows: List[dict] = []
    summary_rows: List[dict] = []
    run_id = 0
    start_time = time.time()

    try:
        for value in experiment.values:
            for seed in SEEDS:
                run_id += 1
                ts_rows, summary = run_single(experiment, value, seed, run_id)
                timeseries_rows.extend(ts_rows)
                summary_rows.append(summary)

                # Live per-run progress with the headline outcome.
                outcome = _describe_outcome(summary)
                print(
                    f" [{run_id:>3}/{total_runs}] {experiment.param_key}={value:<6} "
                    f"seed={seed:<3} -> {outcome}"
                )
    finally:
        # Always restore the baseline, even if something failed mid-sweep.
        config_dict[experiment.param_key] = original_value

    _write_csv(timeseries_path, timeseries_rows)
    _write_csv(summary_path, summary_rows)

    elapsed = time.time() - start_time
    print("-" * 78)
    print(f" Done: {total_runs} runs in {elapsed:.1f}s "
          f"({elapsed / total_runs:.2f}s/run)")
    print(f" Wrote {len(timeseries_rows)} timeseries rows -> {timeseries_path}")
    print(f" Wrote {len(summary_rows)} summary rows     -> {summary_path}")
    print("=" * 78)


def _describe_outcome(summary: dict) -> str:
    """Build a short human-readable outcome string for live progress."""
    if summary["coexistence"]:
        tag = "CO-EXISTENCE"
    elif summary["north_survived"]:
        tag = "North only"
    elif summary["south_survived"]:
        tag = "South only"
    else:
        tag = "GLOBAL EXTINCTION"
    return (
        f"{tag:<17} "
        f"N={summary['final_north_pop']:<3} S={summary['final_south_pop']:<3} "
        f"gt_N={summary['final_avg_gt_north']:.2f} gt_S={summary['final_avg_gt_south']:.2f}"
    )


def _write_csv(path: str, rows: List[dict]) -> None:
    """Write a list of uniform dict rows to CSV (header from the first row)."""
    if not rows:
        print(f" (no rows to write for {path})")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ======================================================================
# CLI ENTRY POINT
# ======================================================================

def main(argv: List[str]) -> int:
    """Parse the experiment name argument and run it."""
    if len(argv) != 2 or argv[1] not in EXPERIMENTS:
        available = ", ".join(EXPERIMENTS.keys())
        print("Usage: python scripts/experiment_runner.py <experiment>")
        print(f"Available experiments: {available}")
        return 1

    run_experiment(EXPERIMENTS[argv[1]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))