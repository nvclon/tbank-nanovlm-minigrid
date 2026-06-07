import gymnasium as gym
import minigrid  # noqa: F401

from src.data_utils.formats import ZERO_SHOT_ACTION_PROMPT
from src.env_utils import randomize_agent
from src.models.nanovlm_policy import NanoVLMPolicy


def main():
    env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
    policy = NanoVLMPolicy(
        checkpoint_path="base",
        output_format="action_only",
        greedy=True,
        max_new_tokens=12,
        prompt_override=ZERO_SHOT_ACTION_PROMPT,
    )

    for seed in [80000, 80001, 80002, 80003, 80004]:
        env.reset(seed=seed)
        randomize_agent(env, seed)

        text = policy.generate_text(env)
        action_id, invalid = policy.act(env)

        print("=" * 60)
        print("seed:", seed)
        print("generated:", repr(text))
        print("action_id:", action_id)
        print("invalid:", invalid)

    env.close()


if __name__ == "__main__":
    main()