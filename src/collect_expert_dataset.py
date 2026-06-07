import argparse
import json
from collections import Counter
from pathlib import Path

from src.env_utils import randomize_agent
import gymnasium as gym
import minigrid  # noqa: F401
from PIL import Image

from src.data_utils.formats import (
    ACTION_ONLY_PROMPT,
    TEXT_ACTION_PROMPT,
    make_action_only_target,
    make_text_action_target,
)
from src.expert.shortest_path_expert import (
    ACTION_ID_TO_NAME,
    expert_action,
    find_goal_pos,
)


def save_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_split(env_id, split, start_seed, num_episodes, output_dir):
    env = gym.make(env_id, render_mode="rgb_array")

    rows = []
    action_counter = Counter()

    image_dir = output_dir / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)

    for episode_idx in range(num_episodes):
        seed = start_seed + episode_idx
        obs, info = env.reset(seed=seed)
        randomize_agent(env, seed)

        for step in range(env.unwrapped.max_steps):
            action_id = int(expert_action(env))
            action_name = ACTION_ID_TO_NAME[action_id]

            if action_name == "done":
                break

            img = env.render()
            image_path = image_dir / f"seed_{seed}_step_{step}.png"
            Image.fromarray(img).save(image_path)

            agent_pos = tuple(int(v) for v in env.unwrapped.agent_pos)
            goal_pos = tuple(int(v) for v in find_goal_pos(env))
            agent_dir = int(env.unwrapped.agent_dir)

            row = {
                "env_id": env_id,
                "split": split,
                "episode_idx": episode_idx,
                "seed": seed,
                "step": step,
                "image_path": str(image_path).replace("\\", "/"),
                "agent_pos": agent_pos,
                "goal_pos": goal_pos,
                "agent_dir": agent_dir,
                "action_id": action_id,
                "action_name": action_name,
                "prompt_action_only": ACTION_ONLY_PROMPT,
                "target_action_only": make_action_only_target(action_name),
                "prompt_text_action": TEXT_ACTION_PROMPT,
                "target_text_action": make_text_action_target(
                    action_name=action_name,
                    agent_pos=agent_pos,
                    goal_pos=goal_pos,
                    agent_dir=agent_dir,
                ),
            }

            rows.append(row)
            action_counter[action_name] += 1

            obs, reward, terminated, truncated, info = env.step(action_id)

            if terminated or truncated:
                break

    env.close()

    jsonl_path = output_dir / f"{split}.jsonl"
    save_jsonl(rows, jsonl_path)

    return {
        "split": split,
        "episodes": num_episodes,
        "transitions": len(rows),
        "actions": dict(action_counter),
        "jsonl_path": str(jsonl_path).replace("\\", "/"),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--output-dir", type=str, default="data/expert")

    parser.add_argument("--train-episodes", type=int, default=10)
    parser.add_argument("--val-episodes", type=int, default=3)
    parser.add_argument("--test-episodes", type=int, default=3)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    summaries.append(
        collect_split(
            env_id=args.env_id,
            split="train",
            start_seed=0,
            num_episodes=args.train_episodes,
            output_dir=output_dir,
        )
    )

    summaries.append(
        collect_split(
            env_id=args.env_id,
            split="val",
            start_seed=10_000,
            num_episodes=args.val_episodes,
            output_dir=output_dir,
        )
    )

    summaries.append(
        collect_split(
            env_id=args.env_id,
            split="test",
            start_seed=20_000,
            num_episodes=args.test_episodes,
            output_dir=output_dir,
        )
    )

    summary_path = output_dir / "summary.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()