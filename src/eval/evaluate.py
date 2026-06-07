import argparse
import json
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401

from collections import Counter
from tqdm import tqdm
from src.expert.shortest_path_expert import ACTION_ID_TO_NAME

from src.models.nanovlm_policy import NanoVLMPolicy
from src.env_utils import randomize_agent
from src.expert.shortest_path_expert import expert_action


class ExpertPolicy:
    def act(self, env):
        return int(expert_action(env)), False


class ForwardPolicy:
    def act(self, env):
        return 2, False


def evaluate_policy(policy, env_id, num_episodes, start_seed, randomize_start=True, max_episode_steps=None):    
    env = gym.make(env_id, render_mode="rgb_array")
    results = []
    action_counter = Counter()

    for episode_idx in tqdm(range(num_episodes), desc="eval episodes"):
        seed = start_seed + episode_idx
        obs, info = env.reset(seed=seed)

        if randomize_start:
            randomize_agent(env, seed)

        total_reward = 0.0
        invalid_count = 0
        episode_limit = max_episode_steps or env.unwrapped.max_steps

        success = False
        final_steps = episode_limit

        for step in range(episode_limit):
            action_id, invalid = policy.act(env)

            action_counter[ACTION_ID_TO_NAME.get(action_id, str(action_id))] += 1

            if invalid:
                invalid_count += 1

            obs, reward, terminated, truncated, info = env.step(action_id)
            total_reward += reward

            if terminated or truncated:
                success = terminated and reward > 0
                final_steps = step + 1
                break

        results.append(
            {
                "seed": seed,
                "success": success,
                "return": total_reward,
                "steps": final_steps,
                "invalid_count": invalid_count,
            }
        )
    env.close()

    success_rate = sum(r["success"] for r in results) / len(results)
    avg_return = sum(r["return"] for r in results) / len(results)
    avg_steps = sum(r["steps"] for r in results) / len(results)
    invalid_rate = sum(r["invalid_count"] for r in results) / sum(r["steps"] for r in results)

    return {
        "env_id": env_id,
        "num_episodes": num_episodes,
        "success_rate": success_rate,
        "avg_return": avg_return,
        "avg_steps": avg_steps,
        "invalid_rate": invalid_rate,
        "action_counts": dict(action_counter),
        "episodes": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument(
        "--policy",
        type=str,
        default="expert",
        choices=["expert", "forward", "nanovlm"],
    )
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=30_000)
    parser.add_argument("--output-path", type=str, default="outputs/eval_metrics.json")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output-format",
        type=str,
        default="action_only",
        choices=["action_only", "text_action"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    
    args = parser.parse_args()

    if args.policy == "expert":
        policy = ExpertPolicy()

    elif args.policy == "forward":
        policy = ForwardPolicy()

    elif args.policy == "nanovlm":
        if args.checkpoint_path is None:
            raise ValueError("--checkpoint-path is required for NanoVLM policy")

        policy = NanoVLMPolicy(
            checkpoint_path=args.checkpoint_path,
            output_format=args.output_format,
            greedy=True,
            max_new_tokens=args.max_new_tokens,
        )

    else:
        raise ValueError(f"Unknown policy: {args.policy}")

    metrics = evaluate_policy(
        policy=policy,
        env_id=args.env_id,
        num_episodes=args.num_episodes,
        start_seed=args.start_seed,
        max_episode_steps=args.max_episode_steps,
        randomize_start=True,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps({k: v for k, v in metrics.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()