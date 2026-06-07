import argparse
import json
from pathlib import Path

from src.data_utils.formats import ZERO_SHOT_ACTION_PROMPT
from src.eval.evaluate import evaluate_policy
from src.models.nanovlm_policy import NanoVLMPolicy


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=30)
    parser.add_argument("--output-path", type=str, default="outputs/main/eval_20/zero_shot_eval_20.json")

    args = parser.parse_args()

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
        num_episodes=args.num_episodes,
        start_seed=30000,
        randomize_start=True,
        max_episode_steps=args.max_episode_steps,
    )

    save_json(args.output_path, metrics)
    print(json.dumps({k: v for k, v in metrics.items() if k != "episodes"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()