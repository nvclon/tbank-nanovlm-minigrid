import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRIC_FILES = {
    "Zero-shot": "zero_shot_eval_20.json",
    "Forward": "forward_eval_20.json",
    "Expert": "expert_eval_20.json",
    "SFT-action": "sft_action_eval_20.json",
    "GRPO-action": "grpo_action_eval_20.json",
    "SFT-text+action": "sft_text_action_eval_20.json",
    "GRPO-text+action": "grpo_text_action_eval_20.json",
}


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def make_summary_table(metrics_dir, figures_dir):
    rows = []

    for method, filename in METRIC_FILES.items():
        data = load_json(metrics_dir / filename)

        rows.append(
            {
                "method": method,
                "success_rate": data["success_rate"],
                "avg_return": data["avg_return"],
                "avg_steps": data["avg_steps"],
                "invalid_rate": data["invalid_rate"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(figures_dir / "summary_table.csv", index=False)

    return df


def plot_final_comparison(df, figures_dir):
    for metric in ["success_rate", "avg_return", "avg_steps", "invalid_rate"]:
        plt.figure(figsize=(9, 4))
        plt.bar(df["method"], df[metric])
        plt.xticks(rotation=30, ha="right")
        plt.ylabel(metric)
        plt.title(f"Final comparison: {metric}")
        plt.tight_layout()
        plt.savefig(figures_dir / f"final_{metric}.png", dpi=180)
        plt.close()


def plot_sft_losses(metrics_dir, figures_dir):
    configs = [
        ("SFT-action", "sft_action_train_metrics.json"),
        ("SFT-text+action", "sft_text_action_train_metrics.json"),
    ]

    plt.figure(figsize=(8, 4))

    for label, filename in configs:
        data = load_json(metrics_dir / filename)
        steps = data.get("steps", [])

        if not steps:
            continue

        xs = [x["step"] for x in steps]
        ys = [x["val_loss"] for x in steps]

        plt.plot(xs, ys, marker="o", label=label)

    plt.xlabel("Training step")
    plt.ylabel("Validation loss")
    plt.title("SFT validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "sft_val_loss.png", dpi=180)
    plt.close()


def plot_grpo_diagnostics(metrics_dir, figures_dir):
    configs = [
        ("GRPO-action", "grpo_action_train_metrics.json"),
        ("GRPO-text+action", "grpo_text_action_train_metrics.json"),
    ]

    for label, filename in configs:
        data = load_json(metrics_dir / filename)
        updates = data.get("updates", [])

        if not updates:
            continue

        xs = [u["update"] for u in updates]
        reward_means = [u.get("reward_mean", 0.0) for u in updates]
        reward_stds = [u.get("reward_std", 0.0) for u in updates]

        plt.figure(figsize=(8, 4))
        plt.plot(xs, reward_means, marker="o")
        plt.xlabel("GRPO update")
        plt.ylabel("Group reward mean")
        plt.title(f"{label}: group reward mean")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{label.lower().replace('+', '_').replace('-', '_')}_reward_mean.png", dpi=180)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(xs, reward_stds, marker="o")
        plt.xlabel("GRPO update")
        plt.ylabel("Group reward std")
        plt.title(f"{label}: group reward std")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{label.lower().replace('+', '_').replace('-', '_')}_reward_std.png", dpi=180)
        plt.close()


def main():
    metrics_dir = Path("results/metrics")
    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = make_summary_table(metrics_dir, figures_dir)
    plot_final_comparison(df, figures_dir)
    plot_sft_losses(metrics_dir, figures_dir)
    plot_grpo_diagnostics(metrics_dir, figures_dir)

    print(df)
    print(f"Saved figures to {figures_dir}")


if __name__ == "__main__":
    main()