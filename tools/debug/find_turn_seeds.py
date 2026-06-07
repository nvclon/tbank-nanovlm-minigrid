import gymnasium as gym
import minigrid  # noqa: F401

from src.env_utils import randomize_agent
from src.expert.shortest_path_expert import expert_action, ACTION_ID_TO_NAME


def main():
    env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")

    found = []

    for seed in range(30000, 30500):
        env.reset(seed=seed)
        randomize_agent(env, seed)

        action_id = int(expert_action(env))
        action_name = ACTION_ID_TO_NAME[action_id]

        if action_name in ["left", "right"]:
            found.append((seed, action_name))

        if len(found) >= 20:
            break

    env.close()

    for seed, action_name in found:
        print(seed, action_name)


if __name__ == "__main__":
    main()