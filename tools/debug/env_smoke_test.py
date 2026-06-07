import gymnasium as gym
import minigrid  # важно: регистрирует MiniGrid environments
from PIL import Image


def main():
    env_id = "MiniGrid-Empty-8x8-v0"

    env = gym.make(env_id, render_mode="rgb_array")
    obs, info = env.reset(seed=42)

    print("Env:", env_id)
    print("Observation keys:", obs.keys())
    print("Action space:", env.action_space)
    print("Mission:", obs.get("mission"))

    # MiniGrid actions:
    # 0 = left, 1 = right, 2 = forward
    actions = [
        env.unwrapped.actions.left,
        env.unwrapped.actions.right,
        env.unwrapped.actions.forward,
    ]

    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"step={i}, action={action}, reward={reward}, "
            f"terminated={terminated}, truncated={truncated}"
        )

        if terminated or truncated:
            break

    img = env.render()
    Image.fromarray(img).save("outputs/smoke_test/minigrid_empty_8x8.png")

    print("Saved image to outputs/smoke_test/minigrid_empty_8x8.png")

    env.close()


if __name__ == "__main__":
    main()