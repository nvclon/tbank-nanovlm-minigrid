import argparse
import json
import textwrap
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import minigrid  # noqa: F401

from src.env_utils import randomize_agent
from src.expert.shortest_path_expert import ACTION_ID_TO_NAME, expert_action
from src.models.action_parser import parse_action_only, parse_text_action
from src.models.nanovlm_policy import NanoVLMPolicy


def wrap_text(text, width=28, max_lines=5):
    text = text.replace("\n", " ")
    lines = textwrap.wrap(text, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] += "..."
    return "\n".join(lines)


def parse_output(text, output_format):
    if output_format == "action_only":
        return parse_action_only(text)
    return parse_text_action(text)


def run_episode(policy, env_id, seed, output_format, max_steps):
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    randomize_agent(env, seed)

    frames = []
    total_reward = 0.0
    success = False

    for step in range(max_steps):
        image = env.render()

        if policy is None:
            action_id = int(expert_action(env))
            generated = ACTION_ID_TO_NAME[action_id]
            invalid = False
        else:
            generated = policy.generate_text(env)
            action_id, invalid = parse_output(generated, output_format)

        expert_id = int(expert_action(env))
        expert_name = ACTION_ID_TO_NAME[expert_id]
        pred_name = ACTION_ID_TO_NAME.get(action_id, str(action_id))

        frames.append(
            {
                "step": step,
                "image": image,
                "generated": generated,
                "action_id": action_id,
                "action_name": pred_name,
                "expert_action_id": expert_id,
                "expert_action_name": expert_name,
                "match_expert": action_id == expert_id,
                "invalid": invalid,
            }
        )

        _, reward, terminated, truncated, _ = env.step(action_id)
        total_reward += reward

        if terminated or truncated:
            success = terminated and reward > 0
            frames.append(
                {
                    "step": step + 1,
                    "image": env.render(),
                    "generated": "<terminal>",
                    "action_id": None,
                    "action_name": None,
                    "expert_action_id": None,
                    "expert_action_name": None,
                    "match_expert": None,
                    "invalid": False,
                }
            )
            break

    env.close()

    return {
        "seed": seed,
        "success": success,
        "return": total_reward,
        "steps": len(frames) - 1,
        "frames": frames,
    }


def save_episode_plot(episode, output_path, output_format):
    frames = episode["frames"]
    n = len(frames)

    cols = min(n, 6)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.7))

    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for i, frame in enumerate(frames):
        ax = axes[i]
        ax.imshow(frame["image"])
        ax.axis("off")

        if frame["generated"] == "<terminal>":
            title = f"step={frame['step']}\nterminal"
        else:
            if output_format == "action_only":
                pred_text = frame["action_name"]
            else:
                pred_text = wrap_text(frame["generated"], width=30, max_lines=4)

            title = (
                f"step={frame['step']} pred={frame['action_name']} "
                f"expert={frame['expert_action_name']}\n"
                f"invalid={frame['invalid']} match={frame['match_expert']}\n"
                f"{pred_text}"
            )

        ax.set_title(title, fontsize=7)

    fig.suptitle(
        f"seed={episode['seed']} | success={episode['success']} | "
        f"return={episode['return']:.3f} | steps={episode['steps']}",
        fontsize=11,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--policy", type=str, default="nanovlm", choices=["nanovlm", "expert"])
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--output-format", type=str, choices=["action_only", "text_action"], required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--start-seed", type=int, default=90_000)
    parser.add_argument("--num-episodes", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=40)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.policy == "expert":
        policy = None
    else:
        if args.checkpoint_path is None:
            raise ValueError("--checkpoint-path is required for nanovlm policy")

        policy = NanoVLMPolicy(
            checkpoint_path=args.checkpoint_path,
            output_format=args.output_format,
            greedy=True,
            max_new_tokens=args.max_new_tokens,
        )

    summary = []

    for i in range(args.num_episodes):
        seed = args.start_seed + i

        episode = run_episode(
            policy=policy,
            env_id=args.env_id,
            seed=seed,
            output_format=args.output_format,
            max_steps=args.max_episode_steps,
        )

        png_path = output_dir / f"trajectory_seed_{seed}.png"
        save_episode_plot(episode, png_path, args.output_format)

        json_episode = {
            "seed": episode["seed"],
            "success": episode["success"],
            "return": episode["return"],
            "steps": episode["steps"],
            "frames": [
                {
                    k: v
                    for k, v in frame.items()
                    if k != "image"
                }
                for frame in episode["frames"]
            ],
        }

        summary.append(json_episode)

        with (output_dir / f"trajectory_seed_{seed}.json").open("w", encoding="utf-8") as f:
            json.dump(json_episode, f, indent=2, ensure_ascii=False)

        print(f"saved: {png_path}")

    with (output_dir / "trajectory_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()