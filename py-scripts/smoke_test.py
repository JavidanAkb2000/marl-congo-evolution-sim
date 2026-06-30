import time
import random
from pettingzoo.classic import tictactoe_v3


def main():
    print("📺 Vizual MARL Pəncərəsi Açılır... Gözləyin...")

    # render_mode="human" təyin etdikdə Pygame pəncərəsi avtomatik açılır

    # screen_height-i kiçilt
    env = tictactoe_v3.env(
        render_mode="human",
        screen_height=500,  # 300, 400, 500 - istədiyini yaz
    )

    env.reset(seed=42)

    print(f"👥 Aktiv Agentlər: {env.agents}")
    print("--- Canlı Oyun Başlayır ---")

    for agent in env.agent_iter():
        observation, reward, terminated, truncated, info = env.last()

        if terminated or truncated:
            action = None
        else:
            # Diqqət: Tic-Tac-Toe-da boş xanaya gediş etmək şərtdir!
            # 'action_mask' bizə hansı xanaların boş (qanuni) olduğunu 1 və 0-larla deyir.
            legal_actions = [i for i, allowed in enumerate(observation["action_mask"]) if allowed]

            # Agent yalnız icazə verilən qanuni xanalardan birini təsadüfi seçir
            action = random.choice(legal_actions) if legal_actions else None

        # Agent hərəkəti mühitə ötürür
        env.step(action)

        # Victus o qədər sürətlidir ki, bura fasilə qoymasaq oyun 0.001 saniyəyə bitər.
        # Hər gedişi gözümüzlə görə bilmək üçün 0.5 saniyəlik süni fasilə veririk.
        time.sleep(0.5)

    env.close()
    print("✅ Canlı simulyasiya pəncərəsi uğurla bağlandı!")


if __name__ == "__main__":
    main()