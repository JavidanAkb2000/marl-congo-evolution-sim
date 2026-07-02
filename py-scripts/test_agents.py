"""
scripts/test_agents.py

Phase 3 verification script.

Runs a 10-step simulated "thinking loop" for two contrasting agent
profiles so we can visually confirm the pure-Python Fuzzy Logic
Controller behaves as expected:

    - North-Bank Style Agent: High G_T (aggressive genotype), starved
      (high hunger), high stress. Expected to lean heavily toward
      ATTACK, especially once a rival is detected.
    - South-Bank Style Agent: High G_E (cooperative genotype), saturated
      (low hunger), low stress. Expected to lean toward MOVE/EXPLORE and
      COOPERATE, rarely toward ATTACK.

Each step feeds alternating dummy sensory inputs (food detected / rival
detected) and prints a clearly formatted breakdown of: current bio-state,
fuzzified membership degrees, raw rule-base scores, normalized action
probabilities, and the action ultimately chosen.

Run with:
    uv run scripts/test_agents.py
"""

from __future__ import annotations

import os
import random
import sys

# Ensure the project root is importable when this script is executed
# directly (e.g. `uv run scripts/test_agents.py`) regardless of the
# current working directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agents.agent import ActionType, EvolvableAgent  # noqa: E402

# ----------------------------------------------------------------------
# Terminal formatting helpers
# ----------------------------------------------------------------------
LINE_WIDTH = 78
SECTION_CHAR = "="
SUBSECTION_CHAR = "-"


def print_section(title: str) -> None:
    print()
    print(SECTION_CHAR * LINE_WIDTH)
    print(f" {title}")
    print(SECTION_CHAR * LINE_WIDTH)


def print_subsection(title: str) -> None:
    print(SUBSECTION_CHAR * LINE_WIDTH)
    print(f" {title}")
    print(SUBSECTION_CHAR * LINE_WIDTH)


def format_membership(label: str, low: float, med: float, high: float) -> str:
    return (
        f"   {label:<7} -> LOW: {low:0.3f}   MEDIUM: {med:0.3f}   HIGH: {high:0.3f}"
    )


def format_action_table(
    raw_weights: dict, probabilities: dict, chosen: ActionType
) -> str:
    lines = []
    header = f"   {'ACTION':<12}{'RAW SCORE':>12}{'PROBABILITY':>15}{'':>4}"
    lines.append(header)
    lines.append("   " + SUBSECTION_CHAR * (LINE_WIDTH - 3))
    for action in [
        ActionType.MOVE,
        ActionType.EAT,
        ActionType.ATTACK,
        ActionType.COOPERATE,
    ]:
        raw = raw_weights.get(action, 0.0)
        prob = probabilities.get(action, 0.0)
        marker = " <== CHOSEN" if action == chosen else ""
        bar = "#" * int(round(prob * 40))
        lines.append(
            f"   {action.value:<12}{raw:>12.3f}{prob:>14.1%}  {bar}{marker}"
        )
    return "\n".join(lines)


def print_agent_state(agent: EvolvableAgent, label: str) -> None:
    print(
        f"   [{label}] id={agent.agent_id}  "
        f"G_E={agent.g_e:0.3f}  G_T={agent.g_t:0.3f}  "
        f"energy={agent.energy:6.2f}  hunger={agent.hunger:6.2f}  "
        f"stress={agent.stress:6.2f}  alive={agent.is_alive}"
    )


# ----------------------------------------------------------------------
# Simulated dummy sensory environment
# ----------------------------------------------------------------------
def generate_dummy_sensory_inputs(step: int, rng: random.Random) -> dict:
    """Produce alternating/randomized dummy food & rival detection flags.

    Alternates food visibility roughly every other step and injects a
    rival sighting with moderate probability, so both agents get
    exercised across a variety of rule-firing conditions over the
    10-step loop.
    """
    food_present = (step % 2 == 0) or (rng.random() < 0.3)
    rival_present = rng.random() < 0.45
    rival_count = rng.randint(1, 3) if rival_present else 0
    return {
        "food_present": food_present,
        "rival_present": rival_present,
        "rival_count": rival_count,
    }


def run_thinking_loop(agent: EvolvableAgent, label: str, steps: int, rng: random.Random) -> None:
    """Execute a `steps`-length metabolic + fuzzy-brain loop for one agent."""
    print_section(f"THINKING LOOP: {label} (Agent #{agent.agent_id})")

    for step in range(1, steps + 1):
        if not agent.is_alive:
            print(f"\n   Step {step:02d}: agent is DEAD. Skipping further ticks.")
            continue

        sensory = generate_dummy_sensory_inputs(step, rng)

        # Update stress from local context (rival proximity + starvation)
        # BEFORE the brain fires, so the fuzzy inputs reflect the agent's
        # current perception of the world this tick.
        agent.update_stress(rival_count=sensory["rival_count"])

        print_subsection(
            f"Step {step:02d}/{steps}  |  "
            f"food_present={sensory['food_present']}  "
            f"rival_present={sensory['rival_present']} "
            f"(count={sensory['rival_count']})"
        )

        print_agent_state(agent, label)

        # Fire the fuzzy emotional brain.
        probabilities = agent.compute_action_weights(
            food_present=sensory["food_present"],
            rival_present=sensory["rival_present"],
        )
        chosen_action = agent.decide_action(
            food_present=sensory["food_present"],
            rival_present=sensory["rival_present"],
        )

        h_low, h_med, h_high = agent.last_fuzzy_hunger
        s_low, s_med, s_high = agent.last_fuzzy_stress

        print()
        print("   Fuzzified Membership Degrees:")
        print(format_membership("Hunger", h_low, h_med, h_high))
        print(format_membership("Stress", s_low, s_med, s_high))
        print()
        print("   Fuzzy Rule Base Output (raw scores -> normalized probabilities):")
        print(format_action_table(agent.last_action_weights, probabilities, chosen_action))
        print()
        print(f"   >>> CHOSEN ACTION: {chosen_action.value}")

        # Apply metabolic cost for the chosen action.
        still_alive = agent.apply_metabolism(chosen_action)

        # Single-item consumption: if the agent chose to EAT and food
        # was actually present, simulate consuming exactly one food
        # item with a region-appropriate energy value.
        if chosen_action == ActionType.EAT and sensory["food_present"] and still_alive:
            region_energy_value = 10.0 if label.startswith("North") else 40.0
            gained = agent.eat_single_food_item(region_energy_value)
            print(
                f"   >>> Consumed ONE food item "
                f"(region value={region_energy_value:.1f}) -> energy +{gained:.2f}"
            )

        if not still_alive:
            print(f"\n   *** Agent #{agent.agent_id} ({label}) has DIED at step {step}. ***")

    print()
    print_agent_state(agent, f"{label} FINAL STATE")


def main() -> None:
    rng = random.Random(7)  # Fixed seed for reproducible demo output.

    print_section("PHASE 3 VERIFICATION: DIGITAL DNA & FUZZY EMOTIONAL BRAIN")
    print(" Comparing a North-Bank (scarcity/aggressive) profile against a")
    print(" South-Bank (abundance/cooperative) profile across 10 simulated steps.")

    # North-Bank Style Agent: high aggression gene, starved, high stress.
    north_agent = EvolvableAgent(
        agent_id=1,
        x=25,
        y=40,
        forced_g_t=0.85,          # High Thanatos (aggression)
        initial_energy=15.0,       # -> hunger = 85 (starved)
        initial_stress=70.0,       # Already under significant stress
        rng=random.Random(101),
    )

    # South-Bank Style Agent: high cooperation gene, saturated, low stress.
    south_agent = EvolvableAgent(
        agent_id=2,
        x=25,
        y=10,
        forced_g_t=0.10,          # Low Thanatos -> high Eros (cooperation)
        initial_energy=92.0,       # -> hunger = 8 (well-fed)
        initial_stress=8.0,        # Calm, safe environment
        rng=random.Random(202),
    )

    run_thinking_loop(north_agent, "North-Bank (Aggressive/Starved)", steps=10, rng=rng)
    run_thinking_loop(south_agent, "South-Bank (Cooperative/Saturated)", steps=10, rng=rng)

    print_section("VERIFICATION COMPLETE")
    print(" North-Bank agent final:")
    print_agent_state(north_agent, "North")
    print(" South-Bank agent final:")
    print_agent_state(south_agent, "South")
    print()


if __name__ == "__main__":
    main()