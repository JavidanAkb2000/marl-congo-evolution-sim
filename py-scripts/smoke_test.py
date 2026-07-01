import time
import random
from pettingzoo.classic import tictactoe_v3


def main():
    print("📺 Visual MARL Window Opening... Please wait...")

    # Setting render_mode="human" automatically opens the Pygame window

    # Reduce screen_height
    env = tictactoe_v3.env(
        render_mode="human",
        screen_height=500,  # 300, 400, 500 - set as desired
    )

    env.reset(seed=42)

    print(f"👥 Active Agents: {env.agents}")
    print("--- Live Game Starting ---")

    for agent in env.agent_iter():
        observation, reward, terminated, truncated, info = env.last()

        if terminated or truncated:
            action = None
        else:
            # Attention: In Tic-Tac-Toe, moving to an empty cell is required!
            # 'action_mask' tells us which cells are empty (legal) using 1s and 0s.
            legal_actions = [i for i, allowed in enumerate(observation["action_mask"]) if allowed]

            # The agent randomly selects only one of the allowed legal cells
            action = random.choice(legal_actions) if legal_actions else None

        # Agent sends the action to the environment
        env.step(action)

        # The execution is so fast that without a pause, the game would end in 0.001 seconds.
        # We add a 0.5-second artificial delay to see each move with our own eyes.
        time.sleep(0.5)

    env.close()
    print("✅ Live simulation window closed successfully!")


if __name__ == "__main__":
    main()
