import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np

from src.expert.shortest_path_expert import ACTION_ID_TO_NAME, expert_action
from src.models.action_sampler import sample_action
from src.env_utils import randomize_agent
from src.models.nanovlm_policy import NanoVLMPolicy
from src.expert.shortest_path_expert import ACTION_ID_TO_NAME


def rollout_from_first_action(env_id, seed, first_action, max_steps=30):
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    randomize_agent(env, seed)

    total_reward = 0.0
    steps = 0

    obs, reward, terminated, truncated, info = env.step(first_action)
    total_reward += reward
    steps += 1

    if terminated or truncated:
        env.close()
        return total_reward, terminated and reward > 0, steps

    for _ in range(max_steps - 1):
        action = int(expert_action(env))

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            env.close()
            return total_reward, terminated and reward > 0, steps

    env.close()
    return total_reward, False, steps


def main():
    env_id = "MiniGrid-Empty-8x8-v0"
    seed = 30001
    group_size = 8

    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    randomize_agent(env, seed)

    policy = NanoVLMPolicy(
        checkpoint_path="outputs/sft_action_shifted_single512/final",
        output_format="action_only",
        greedy=False,
        max_new_tokens=8,
    )

    samples = []

    for i in range(group_size):
        sample = sample_action(policy, env, temperature=2.0)
        action_id = sample["action_id"]
        invalid = False

        ret, success, steps = rollout_from_first_action(
            env_id=env_id,
            seed=seed,
            first_action=action_id,
            max_steps=30,
        )

        samples.append(
            {
                "sample": i,
                "action_id": action_id,
                "action_name": ACTION_ID_TO_NAME[action_id],
                "invalid": invalid,
                "return": ret,
                "success": success,
                "steps": steps,
                "logprob": float(sample["logprob"]),
                "prob": float(sample["prob"]),
            }
        )

    env.close()

    rewards = np.array([s["return"] for s in samples], dtype=np.float32)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    print("Samples:")
    for s, adv in zip(samples, advantages):
        print(
            f"{s['sample']:02d} | "
            f"action={s['action_name']:<7} | "
            f"prob={s['prob']:.3f} | "
            f"logprob={s['logprob']:.3f} | "
            f"return={s['return']:.3f} | "
            f"success={s['success']} | "
            f"steps={s['steps']:<2} | "
            f"adv={adv:.3f}"
        )

    print()
    print("reward mean:", float(rewards.mean()))
    print("reward std:", float(rewards.std()))


if __name__ == "__main__":
    main()