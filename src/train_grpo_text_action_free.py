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
from src.models.action_parser import parse_text_action
from src.models.action_sampler import build_prompt
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


def sample_completion(policy, env, max_new_tokens, temperature):
    prefix_text, images_tensor = build_prompt(policy, env)

    encoded = policy.tokenizer(
        prefix_text,
        return_tensors="pt",
        truncation=True,
        max_length=policy.cfg.lm_max_length,
    )

    input_ids = encoded["input_ids"].to(policy.device)
    attention_mask = encoded["attention_mask"].to(policy.device)

    with torch.no_grad():
        generated_ids = policy.model.generate(
            input_ids=input_ids,
            images=[images_tensor],
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            top_k=50,
            top_p=0.9,
            temperature=temperature,
            greedy=False,
        )

    generated_text = policy.tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )

    action_id, invalid = parse_text_action(generated_text)

    return {
        "prefix_text": prefix_text,
        "images_tensor": images_tensor.cpu(),
        "generated_ids": generated_ids[0].detach().cpu(),
        "generated_text": generated_text,
        "action_id": action_id,
        "invalid": invalid,
    }


def completion_logprob(policy, sample):
    prefix_ids = policy.tokenizer(
        sample["prefix_text"],
        return_tensors="pt",
        truncation=True,
        max_length=policy.cfg.lm_max_length,
    )["input_ids"].to(policy.device)

    generated_ids = sample["generated_ids"].unsqueeze(0).to(policy.device)

    input_ids = torch.cat([prefix_ids, generated_ids], dim=1)
    input_ids = input_ids[:, : policy.cfg.lm_max_length]

    attention_mask = torch.ones_like(input_ids)

    labels = torch.full_like(input_ids, -100)

    prefix_len = prefix_ids.shape[1]
    seq_len = input_ids.shape[1]

    if prefix_len < seq_len:
        labels[:, prefix_len - 1 : seq_len - 1] = input_ids[:, prefix_len:seq_len]

    logits, _ = policy.model(
        input_ids=input_ids,
        images=[sample["images_tensor"]],
        attention_mask=attention_mask,
        targets=None,
    )

    logits = policy.model.decoder.head(logits)
    log_probs = torch.log_softmax(logits, dim=-1)

    mask = labels != -100
    token_log_probs = log_probs.gather(
        -1,
        labels.clamp_min(0).unsqueeze(-1),
    ).squeeze(-1)

    return token_log_probs[mask].sum()


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


def evaluate_free_text_policy(policy, env_id, num_episodes, start_seed, max_episode_steps, max_new_tokens):
    results = []
    invalid_count = 0

    for episode_idx in range(num_episodes):
        seed = start_seed + episode_idx

        env = gym.make(env_id, render_mode="rgb_array")
        env.reset(seed=seed)
        randomize_agent(env, seed)

        total_reward = 0.0
        success = False
        final_steps = max_episode_steps

        for step in range(max_episode_steps):
            sample = sample_completion(
                policy=policy,
                env=env,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
            )

            if sample["invalid"]:
                invalid_count += 1

            obs, reward, terminated, truncated, info = env.step(sample["action_id"])
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

    total_steps = sum(r["steps"] for r in results)

    return {
        "success_rate": sum(r["success"] for r in results) / len(results),
        "avg_return": sum(r["return"] for r in results) / len(results),
        "avg_steps": total_steps / len(results),
        "invalid_rate": invalid_count / max(total_steps, 1),
    }


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="outputs/sft_text_action_decoder_last_lr1e5_single512/final",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/grpo_text_action_free_debug")

    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--train-mode", type=str, default="mp", choices=["mp", "decoder_last"])

    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--invalid-penalty", type=float, default=0.05)

    parser.add_argument("--max-episode-steps", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=5)

    parser.add_argument("--start-seed", type=int, default=60_000)
    parser.add_argument("--eval-start-seed", type=int, default=70_000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = NanoVLMPolicy(
        checkpoint_path=args.checkpoint_path,
        output_format="text_action",
        greedy=False,
        max_new_tokens=args.max_new_tokens,
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

    for update_idx in tqdm(range(1, args.updates + 1), desc="grpo text updates"):
        seed = args.start_seed + update_idx

        env = gym.make(args.env_id, render_mode="rgb_array")
        env.reset(seed=seed)
        randomize_agent(env, seed)

        samples = []

        for _ in range(args.group_size):
            sample = sample_completion(
                policy=policy,
                env=env,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            reward, success, steps = rollout_after_first_action(
                env_id=args.env_id,
                seed=seed,
                first_action=sample["action_id"],
                max_steps=args.max_episode_steps,
            )

            if sample["invalid"]:
                reward -= args.invalid_penalty

            sample["reward"] = reward
            sample["success"] = success
            sample["steps"] = steps

            samples.append(sample)

        env.close()

        rewards = np.array([s["reward"] for s in samples], dtype=np.float32)

        reward_mean = float(rewards.mean())
        reward_std = float(rewards.std())

        row = {
            "update": update_idx,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "actions": [int(s["action_id"]) for s in samples],
            "invalid": [bool(s["invalid"]) for s in samples],
            "texts": [s["generated_text"] for s in samples],
        }

        if reward_std < 1e-8:
            row["loss"] = 0.0
            row["skipped"] = True
            metrics["updates"].append(row)
            save_json(output_dir / "metrics.json", metrics)
            continue

        advantages = (rewards - reward_mean) / (reward_std + 1e-8)

        optimizer.zero_grad(set_to_none=True)

        loss = None

        for sample, adv in zip(samples, advantages):
            lp = completion_logprob(policy, sample)
            sample_loss = -float(adv) * lp

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

        row["loss"] = float(loss.detach().cpu())
        row["group_success"] = sum(s["success"] for s in samples) / len(samples)
        row["group_avg_steps"] = sum(s["steps"] for s in samples) / len(samples)

        metrics["updates"].append(row)

        if update_idx % args.eval_every == 0:
            eval_metrics = evaluate_free_text_policy(
                policy=policy,
                env_id=args.env_id,
                num_episodes=args.eval_episodes,
                start_seed=args.eval_start_seed,
                max_episode_steps=args.max_episode_steps,
                max_new_tokens=args.max_new_tokens,
            )
            eval_metrics["update"] = update_idx
            metrics["eval"].append(eval_metrics)

            print()
            print("train:", {k: v for k, v in row.items() if k != "texts"})
            print("sample text:", row["texts"][0])
            print("eval:", eval_metrics)

        save_json(output_dir / "metrics.json", metrics)

    policy.model.save_pretrained(str(output_dir / "final"))
    save_json(output_dir / "metrics.json", metrics)

    print("saved final checkpoint:", output_dir / "final")


if __name__ == "__main__":
    main()