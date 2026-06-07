import argparse
import textwrap
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import minigrid  # noqa: F401

from src.env_utils import randomize_agent
from src.models.action_parser import parse_action_only, parse_text_action
from src.models.nanovlm_policy import NanoVLMPolicy


def wrap_text(text, width=42, max_lines=8):
    text = text.replace("\n", " ")
    lines = textwrap.wrap(text, width=width)
    lines = lines[:max_lines]

    if len(textwrap.wrap(text, width=width)) > max_lines:
        lines[-1] = lines[-1] + "..."

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--output-format", type=str, choices=["action_only", "text_action"], required=True)
    parser.add_argument("--output-path", type=str, required=True)

    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--start-seed", type=int, default=90_000)
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=40)

    args = parser.parse_args()

    policy = NanoVLMPolicy(
        checkpoint_path=args.checkpoint_path,
        output_format=args.output_format,
        greedy=True,
        max_new_tokens=args.max_new_tokens,
    )

    env = gym.make(args.env_id, render_mode="rgb_array")

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(args.num_examples):
        seed = args.start_seed + i

        env.reset(seed=seed)
        randomize_agent(env, seed)

        image = env.render()
        generated = policy.generate_text(env)

        if args.output_format == "action_only":
            action_id, invalid = parse_action_only(generated)
        else:
            action_id, invalid = parse_text_action(generated)

        axes[i].imshow(image)
        axes[i].axis("off")

        title = (
            f"seed={seed}, action_id={action_id}, invalid={invalid}\n"
            f"{wrap_text(generated)}"
        )

        axes[i].set_title(title, fontsize=8)

    env.close()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()