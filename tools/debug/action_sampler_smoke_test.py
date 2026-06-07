import gymnasium as gym
import minigrid  # noqa: F401

from src.env_utils import randomize_agent
from src.models.action_sampler import action_distribution, sample_action
from src.models.nanovlm_policy import NanoVLMPolicy


def main():
    env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
    env.reset(seed=30001)
    randomize_agent(env, seed=30001)

    policy = NanoVLMPolicy(
        checkpoint_path="outputs/sft_action_shifted_single512/final",
        output_format="action_only",
        greedy=True,
        max_new_tokens=8,
    )

    dist = action_distribution(policy, env, temperature=1.0)

    print("Distribution:")
    for action, lp, p in zip(dist["actions"], dist["logprobs"], dist["probs"]):
        print(f"{action:<7} logprob={float(lp):.3f} prob={float(p):.4f}")

    print()
    print("Samples:")
    for _ in range(10):
        s = sample_action(policy, env, temperature=2.0)
        print(s["action"], float(s["prob"]))

    env.close()


if __name__ == "__main__":
    main()