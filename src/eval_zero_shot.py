from src.data_utils.formats import ZERO_SHOT_ACTION_PROMPT
from src.eval.evaluate import evaluate_policy
from src.models.nanovlm_policy import NanoVLMPolicy


def main():
    policy = NanoVLMPolicy(
        checkpoint_path="base",
        output_format="action_only",
        greedy=True,
        max_new_tokens=12,
        prompt_override=ZERO_SHOT_ACTION_PROMPT,
    )

    metrics = evaluate_policy(
        policy=policy,
        env_id="MiniGrid-Empty-8x8-v0",
        num_episodes=20,
        start_seed=80_000,
        randomize_start=True,
        max_episode_steps=30,
    )

    print({k: v for k, v in metrics.items() if k != "episodes"})


if __name__ == "__main__":
    main()