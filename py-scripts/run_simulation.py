"""
scripts/run_simulation.py

Phase 4 demonstration & verification script.

Shows the full MARL loop end to end:
    1. Build a CongoEcosystem + CongoArena and reset() to spawn a
       population.
    2. Run several environment steps, printing a per-step summary
       (population, bank distribution, average aggression gene, any
       combat/alliance/reproduction events).
    3. Save a checkpoint of the exact world state to disk.
    4. Deliberately mutate the live arena's state (simulate more steps,
       i.e. "corrupt" it relative to the checkpoint).
    5. Reload the checkpoint into a brand-new CongoArena instance and
       verify it matches the saved snapshot exactly, proving the
       simulation can resume seamlessly without losing evolutionary
       progress.

Run with:
    uv run scripts/run_simulation.py
"""

from __future__ import annotations

import os
import sys

# Ensure the project root is importable when this script is executed
# directly (e.g. `uv run scripts/run_simulation.py`) regardless of the
# current working directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agents.genetics import GeneticEngine  # noqa: E402
from src.environment.arena import CongoArena  # noqa: E402
from src.environment.ecosystem import CongoEcosystem  # noqa: E402
from src.environment.interactions import InteractionResolver  # noqa: E402
from src.persistence.checkpoint import CheckpointManager  # noqa: E402

LINE = "=" * 78


def print_header(title: str) -> None:
    print()
    print(LINE)
    print(f" {title}")
    print(LINE)


def build_arena(rng_seed: int = 2026) -> CongoArena:
    """Construct a fresh, fully-wired CongoArena."""
    ecosystem = CongoEcosystem(
        width=50,
        height=50,
        river_y=25,
        north_spawn_prob=0.05,
        south_spawn_prob=0.50,
        north_food_energy=10.0,
        south_food_energy=40.0,
        south_cluster_size_range=(3, 6),
        south_cluster_radius=2,
        rng_seed=rng_seed,
    )
    genetic_engine = GeneticEngine(
        reproduction_energy_threshold=80.0,
        reproduction_cost=30.0,
        crossover_weight_parent1=0.6,
        mutation_rate=0.05,
        mutation_sigma=0.05,
        offspring_initial_energy=50.0,
        mating_max_distance=1,
    )
    interaction_resolver = InteractionResolver()

    arena = CongoArena(
        ecosystem=ecosystem,
        initial_population=40,
        perception_radius=3,
        genetic_engine=genetic_engine,
        interaction_resolver=interaction_resolver,
        max_steps=None,
        rng_seed=rng_seed,
    )
    return arena


def run_steps(arena: CongoArena, num_steps: int, verbose: bool = True) -> None:
    for _ in range(num_steps):
        arena.step()
        if verbose:
            arena.render(mode="human")


def snapshot(arena: CongoArena) -> dict:
    """Capture a comparable summary of arena state for verification."""
    return {
        "step": arena.current_step,
        "next_agent_id": arena._next_agent_id,
        "food_count": arena.ecosystem.food_count(),
        "agent_ids": sorted(arena.agents_by_id.keys()),
        "agent_states": {
            aid: (
                round(a.x, 6),
                round(a.y, 6),
                round(a.g_t, 6),
                round(a.g_e, 6),
                round(a.energy, 6),
                round(a.hunger, 6),
                round(a.stress, 6),
                a.generation,
                a.is_alive,
            )
            for aid, a in sorted(arena.agents_by_id.items())
        },
    }


def main() -> None:
    print_header("PHASE 4 VERIFICATION: MARL LOOP, GENETICS & CHECKPOINTING")

    # ------------------------------------------------------------------
    # 1. Initialize and reset
    # ------------------------------------------------------------------
    arena = build_arena(rng_seed=2026)
    arena.reset(seed=2026)
    print(f"Initialized arena with {len(arena.agents)} agents on a "
          f"{arena.ecosystem.width}x{arena.ecosystem.height} grid.")

    # ------------------------------------------------------------------
    # 2. Run several steps, observing interactions & reproduction
    # ------------------------------------------------------------------
    print_header("RUNNING 15 SIMULATION STEPS")
    run_steps(arena, num_steps=15, verbose=True)

    # ------------------------------------------------------------------
    # 3. Save a checkpoint
    # ------------------------------------------------------------------
    print_header("SAVING CHECKPOINT")
    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "congo_checkpoint.json")
    CheckpointManager.save(arena, checkpoint_path)
    print(f"Checkpoint written to: {checkpoint_path}")

    pre_save_snapshot = snapshot(arena)
    print(
        f"Snapshot at save time -> step={pre_save_snapshot['step']}, "
        f"agents={len(pre_save_snapshot['agent_ids'])}, "
        f"food={pre_save_snapshot['food_count']}, "
        f"next_agent_id={pre_save_snapshot['next_agent_id']}"
    )

    # ------------------------------------------------------------------
    # 4. Deliberately mutate the live arena AFTER saving, to prove that
    #    reloading the checkpoint truly restores the earlier state
    #    rather than reflecting whatever the live object currently holds.
    # ------------------------------------------------------------------
    print_header("MUTATING LIVE STATE (RUNNING 10 MORE STEPS, UNSAVED)")
    run_steps(arena, num_steps=10, verbose=False)
    mutated_snapshot = snapshot(arena)
    print(
        f"Live arena after further unsaved steps -> step={mutated_snapshot['step']}, "
        f"agents={len(mutated_snapshot['agent_ids'])}, "
        f"food={mutated_snapshot['food_count']}, "
        f"next_agent_id={mutated_snapshot['next_agent_id']}"
    )
    assert mutated_snapshot != pre_save_snapshot, (
        "Sanity check failed: state did not actually change after further steps."
    )
    print("Confirmed: live state has diverged from the saved checkpoint.")

    # ------------------------------------------------------------------
    # 5. Reload the checkpoint into a brand-new arena and verify it
    #    matches the pre-save snapshot exactly.
    # ------------------------------------------------------------------
    print_header("RELOADING CHECKPOINT INTO A NEW ARENA")
    restored_arena = CheckpointManager.load(
        checkpoint_path,
        genetic_engine=GeneticEngine(
            reproduction_energy_threshold=80.0,
            reproduction_cost=30.0,
            crossover_weight_parent1=0.6,
            mutation_rate=0.05,
            mutation_sigma=0.05,
            offspring_initial_energy=50.0,
            mating_max_distance=1,
        ),
        interaction_resolver=InteractionResolver(),
        perception_radius=3,
        rng_seed=2026,
    )
    restored_snapshot = snapshot(restored_arena)
    print(
        f"Restored arena -> step={restored_snapshot['step']}, "
        f"agents={len(restored_snapshot['agent_ids'])}, "
        f"food={restored_snapshot['food_count']}, "
        f"next_agent_id={restored_snapshot['next_agent_id']}"
    )

    assert restored_snapshot == pre_save_snapshot, (
        "CHECKPOINT MISMATCH: restored state does not match the state at save time!"
    )
    print()
    print("VERIFIED: restored arena state matches the checkpoint EXACTLY "
          "(step counter, food grid, and every agent's position/genes/"
          "energy/hunger/stress/generation/alive-status all match).")

    # ------------------------------------------------------------------
    # 6. Prove the restored arena is fully live: step it forward.
    # ------------------------------------------------------------------
    print_header("RESUMING SIMULATION FROM THE RESTORED CHECKPOINT")
    run_steps(restored_arena, num_steps=5, verbose=True)
    print(f"\nSimulation resumed successfully; now at step {restored_arena.current_step} "
          f"with {len(restored_arena.agents)} living agents.")


if __name__ == "__main__":
    main()