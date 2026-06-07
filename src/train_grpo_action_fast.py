import argparse
import json
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch
from tqdm import tqdm

from src.env_utils import randomize_agent
from src.expert.shortest_path_expert import expert_action
from src.models.action_sampler import sample_action, score_action
from src.models.nanovlm_policy import NanoVLMPolicy
from src.train_grpo_action import (
    set_trainable_params,
    count_trainable_params,
    evaluate_constrained_policy,
)


def rollout_after_first_action(env_id, seed, first_action, max_steps):
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    randomize_agent(env, seed)

    total_reward = 0.0
    steps = 0
    success = False

    obs, reward, terminated, truncated, info = env.step(first_action)
    total_reward += reward
    steps += 1

    if terminated or truncated:
        success = terminated and reward > 0
        env.close()
        return total_reward, success, steps

    for _ in range(max_steps - 1):
        action = int(expert_action(env))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            success = terminated and reward > 0
            break

    env.close()
    return total_reward, success, steps


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint-path", type=str, default="outputs/sft_action_shifted_single512/final")
    parser.add_argument("--output-dir", type=str, default="outputs/grpo_action_fast")

    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--train-mode", type=str, default="mp", choices=["mp", "decoder_last"])

    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-episode-steps", type=int, default=30)

    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=25)

    parser.add_argument("--start-seed", type=int, default=40_000)
    parser.add_argument("--eval-start-seed", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = NanoVLMPolicy(
        checkpoint_path=args.checkpoint_path,
        output_format="action_only",
        greedy=False,
        max_new_tokens=8,
    )

    set_trainable_params(policy.model, args.train_mode)
    policy.model.train()

    optimizer = torch.optim.AdamW(
        [p for p in policy.model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    metrics = {
        "args": vars(args),
        "trainable_params": count_trainable_params(policy.model),
        "updates": [],
        "eval": [],
    }

    print("device:", policy.device)
    print("trainable params:", metrics["trainable_params"])
    print("output dir:", output_dir)

    for update_idx in tqdm(range(1, args.updates + 1), desc="grpo fast updates"):
        seed = args.start_seed + update_idx

        env = gym.make(args.env_id, render_mode="rgb_array")
        env.reset(seed=seed)
        randomize_agent(env, seed)

        samples = []

        for _ in range(args.group_size):
            sample = sample_action(policy, env, temperature=args.temperature)

            reward, success, steps = rollout_after_first_action(
                env_id=args.env_id,
                seed=seed,
                first_action=sample["action_id"],
                max_steps=args.max_episode_steps,
            )

            samples.append(
                {
                    "prefix_text": sample["dist"],
                    "action": sample["action"],
                    "action_id": sample["action_id"],
                    "reward": reward,
                    "success": success,
                    "steps": steps,
                }
            )

        env.close()

        rewards = np.array([s["reward"] for s in samples], dtype=np.float32)

        reward_mean = float(rewards.mean())
        reward_std = float(rewards.std())

        if reward_std < 1e-8:
            metrics["updates"].append(
                {
                    "update": update_idx,
                    "reward_mean": reward_mean,
                    "reward_std": reward_std,
                    "loss": 0.0,
                    "skipped": True,
                }
            )
            continue

        advantages = (rewards - reward_mean) / (reward_std + 1e-8)

        optimizer.zero_grad(set_to_none=True)

        loss = None

        for sample, adv in zip(samples, advantages):
            # rebuild same env state to get same prompt/image
            env = gym.make(args.env_id, render_mode="rgb_array")
            env.reset(seed=seed)
            randomize_agent(env, seed)

            from src.models.action_sampler import build_prompt

            prefix_text, images_tensor = build_prompt(policy, env)
            env.close()

            logprob = score_action(
                policy,
                prefix_text,
                images_tensor,
                sample["action"],
                requires_grad=True,
            )

            sample_loss = -float(adv) * logprob

            if loss is None:
                loss = sample_loss
            else:
                loss = loss + sample_loss

        loss = loss / args.group_size
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in policy.model.parameters() if p.requires_grad],
            1.0,
        )

        optimizer.step()
        torch.cuda.empty_cache()

        row = {
            "update": update_idx,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "loss": float(loss.detach().cpu()),
            "group_success": sum(s["success"] for s in samples) / len(samples),
            "group_avg_steps": sum(s["steps"] for s in samples) / len(samples),
            "actions": [s["action"] for s in samples],
        }

        metrics["updates"].append(row)

        if update_idx % args.eval_every == 0:
            eval_metrics = evaluate_constrained_policy(
                policy=policy,
                env_id=args.env_id,
                num_episodes=args.eval_episodes,
                start_seed=args.eval_start_seed,
                max_episode_steps=args.max_episode_steps,
            )

            eval_metrics["update"] = update_idx
            metrics["eval"].append(eval_metrics)

            print()
            print("train:", row)
            print("eval:", eval_metrics)

        if update_idx % args.save_every == 0:
            policy.model.save_pretrained(str(output_dir / f"update_{update_idx}"))

        save_json(output_dir / "metrics.json", metrics)

    policy.model.save_pretrained(str(output_dir / "final"))
    save_json(output_dir / "metrics.json", metrics)

    print("saved final checkpoint:", output_dir / "final")


if __name__ == "__main__":
    main()