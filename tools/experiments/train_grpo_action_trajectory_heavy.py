import argparse
import json
from collections import Counter
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch
from tqdm import tqdm

from src.env_utils import randomize_agent
from src.expert.shortest_path_expert import ACTION_ID_TO_NAME
from src.models.action_sampler import ACTIONS, ACTION_TO_ID, build_prompt, score_action
from src.models.nanovlm_policy import NanoVLMPolicy


def set_trainable_params(model, mode):
    for p in model.parameters():
        p.requires_grad = False

    if mode == "mp":
        for p in model.MP.parameters():
            p.requires_grad = True

    elif mode == "decoder_last":
        for p in model.MP.parameters():
            p.requires_grad = True

        for block in model.decoder.blocks[-1:]:
            for p in block.parameters():
                p.requires_grad = True

    else:
        raise ValueError(f"Unknown train mode: {mode}")


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sample_constrained_action(policy, env, temperature):
    prefix_text, images_tensor = build_prompt(policy, env)

    with torch.no_grad():
        scores = []
        for action in ACTIONS:
            lp = score_action(
                policy,
                prefix_text,
                images_tensor,
                action,
                requires_grad=False,
            )
            scores.append(lp)

        scores = torch.stack(scores)
        probs = torch.softmax(scores / temperature, dim=0)
        idx = torch.multinomial(probs.detach().cpu(), num_samples=1).item()

    action = ACTIONS[idx]
    action_id = ACTION_TO_ID[action]

    return {
        "action": action,
        "action_id": action_id,
        "prefix_text": prefix_text,
        "images_tensor": images_tensor.cpu(),
        "prob": float(probs[idx].detach().cpu()),
        "logprob": float(scores[idx].detach().cpu()),
    }


def rollout_trajectory(policy, env_id, seed, temperature, max_episode_steps):
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    randomize_agent(env, seed)

    total_reward = 0.0
    states = []
    action_counter = Counter()
    success = False
    final_steps = max_episode_steps

    for step in range(max_episode_steps):
        sample = sample_constrained_action(policy, env, temperature)

        states.append(
            {
                "prefix_text": sample["prefix_text"],
                "images_tensor": sample["images_tensor"],
                "action": sample["action"],
            }
        )

        action_counter[ACTION_ID_TO_NAME[sample["action_id"]]] += 1

        obs, reward, terminated, truncated, info = env.step(sample["action_id"])
        total_reward += reward

        if terminated or truncated:
            success = terminated and reward > 0
            final_steps = step + 1
            break

    env.close()

    return {
        "reward": total_reward,
        "success": success,
        "steps": final_steps,
        "states": states,
        "action_counts": dict(action_counter),
    }


def compute_trajectory_logprob(policy, trajectory):
    total_logprob = None

    for state in trajectory["states"]:
        lp = score_action(
            policy,
            state["prefix_text"],
            state["images_tensor"],
            state["action"],
            requires_grad=True,
        )

        if total_logprob is None:
            total_logprob = lp
        else:
            total_logprob = total_logprob + lp

    return total_logprob


def evaluate_constrained_policy(policy, env_id, num_episodes, start_seed, max_episode_steps):
    results = []
    action_counter = Counter()

    for episode_idx in range(num_episodes):
        seed = start_seed + episode_idx
        env = gym.make(env_id, render_mode="rgb_array")
        env.reset(seed=seed)
        randomize_agent(env, seed)

        total_reward = 0.0
        success = False
        final_steps = max_episode_steps

        for step in range(max_episode_steps):
            prefix_text, images_tensor = build_prompt(policy, env)

            scores = []
            for action in ACTIONS:
                lp = score_action(
                    policy,
                    prefix_text,
                    images_tensor,
                    action,
                    requires_grad=False,
                )
                scores.append(lp)

            scores = torch.stack(scores)
            idx = torch.argmax(scores).item()

            action = ACTIONS[idx]
            action_id = ACTION_TO_ID[action]

            action_counter[action] += 1

            obs, reward, terminated, truncated, info = env.step(action_id)
            total_reward += reward

            if terminated or truncated:
                success = terminated and reward > 0
                final_steps = step + 1
                break

        env.close()

        results.append(
            {
                "seed": seed,
                "success": success,
                "return": total_reward,
                "steps": final_steps,
            }
        )

    return {
        "success_rate": sum(r["success"] for r in results) / len(results),
        "avg_return": sum(r["return"] for r in results) / len(results),
        "avg_steps": sum(r["steps"] for r in results) / len(results),
        "action_counts": dict(action_counter),
    }


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint-path", type=str, default="outputs/sft_action_shifted_single512/final")
    parser.add_argument("--output-dir", type=str, default="outputs/grpo_action_debug")

    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--train-mode", type=str, default="mp", choices=["mp", "decoder_last"])

    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-episode-steps", type=int, default=30)

    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=10)

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

    for update_idx in tqdm(range(1, args.updates + 1), desc="grpo updates"):
        group_seed = args.start_seed + update_idx

        trajectories = []
        for _ in range(args.group_size):
            traj = rollout_trajectory(
                policy=policy,
                env_id=args.env_id,
                seed=group_seed,
                temperature=args.temperature,
                max_episode_steps=args.max_episode_steps,
            )
            trajectories.append(traj)

        rewards = np.array([t["reward"] for t in trajectories], dtype=np.float32)

        reward_mean = float(rewards.mean())
        reward_std = float(rewards.std())

        if reward_std < 1e-8:
            advantages = np.zeros_like(rewards)
        else:
            advantages = (rewards - reward_mean) / (reward_std + 1e-8)

        optimizer.zero_grad(set_to_none=True)

        loss = None

        for traj, adv in zip(trajectories, advantages):
            traj_logprob = compute_trajectory_logprob(policy, traj)
            traj_loss = -float(adv) * traj_logprob

            if loss is None:
                loss = traj_loss
            else:
                loss = loss + traj_loss

        loss = loss / args.group_size
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in policy.model.parameters() if p.requires_grad],
            1.0,
        )

        optimizer.step()

        group_success = sum(t["success"] for t in trajectories) / len(trajectories)
        group_steps = sum(t["steps"] for t in trajectories) / len(trajectories)

        row = {
            "update": update_idx,
            "seed": group_seed,
            "loss": float(loss.detach().cpu()),
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "group_success": group_success,
            "group_avg_steps": group_steps,
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
            ckpt_dir = output_dir / f"update_{update_idx}"
            policy.model.save_pretrained(str(ckpt_dir))

        save_json(output_dir / "metrics.json", metrics)

    policy.model.save_pretrained(str(output_dir / "final"))
    save_json(output_dir / "metrics.json", metrics)

    print("saved final checkpoint:", output_dir / "final")


if __name__ == "__main__":
    main()