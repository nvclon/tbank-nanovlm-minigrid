#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="outputs/main"
METRICS_DIR="results/metrics"
TRAJECTORIES_DIR="results/trajectories"

mkdir -p "$METRICS_DIR"
mkdir -p results/figures

cp "$RUN_DIR"/eval_20/*.json "$METRICS_DIR"/

cp "$RUN_DIR/sft_action/metrics.json" "$METRICS_DIR/sft_action_train_metrics.json"
cp "$RUN_DIR/sft_text_action/metrics.json" "$METRICS_DIR/sft_text_action_train_metrics.json"
cp "$RUN_DIR/grpo_action/metrics.json" "$METRICS_DIR/grpo_action_train_metrics.json"
cp "$RUN_DIR/grpo_text_action/metrics.json" "$METRICS_DIR/grpo_text_action_train_metrics.json"

rm -rf "$TRAJECTORIES_DIR"
mkdir -p "$TRAJECTORIES_DIR"

cp -r "$RUN_DIR/trajectories/expert" "$TRAJECTORIES_DIR/expert"
cp -r "$RUN_DIR/trajectories/sft_action" "$TRAJECTORIES_DIR/sft_action"
cp -r "$RUN_DIR/trajectories/grpo_action" "$TRAJECTORIES_DIR/grpo_action"
cp -r "$RUN_DIR/trajectories/sft_text_action" "$TRAJECTORIES_DIR/sft_text_action"
cp -r "$RUN_DIR/trajectories/grpo_text_action" "$TRAJECTORIES_DIR/grpo_text_action"

python -m src.eval.make_result_figures

echo "Exported final metrics, trajectories and figures to results/"
