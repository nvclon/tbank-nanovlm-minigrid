import gymnasium as gym
import minigrid  # noqa: F401

from src.env_utils import randomize_agent
from src.models.nanovlm_policy import NanoVLMPolicy


def main():
    env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
    env.reset(seed=30000)
    randomize_agent(env, seed=30000)

    policy = NanoVLMPolicy(
        checkpoint_path="outputs/sft_text_action_decoder_last_lr1e5_single512/final",
        output_format="text_action",
        greedy=True,
        max_new_tokens=80,
    )

    for i in range(5):
        text = policy.generate_text(env)
        action_id, invalid = policy.act(env)

        print("=" * 60)
        print("generated:", repr(text))
        print("action_id:", action_id)
        print("invalid:", invalid)

        obs, reward, terminated, truncated, info = env.step(action_id)

        if terminated or truncated:
            print("done", reward)
            break

    env.close()


if __name__ == "__main__":
    main()