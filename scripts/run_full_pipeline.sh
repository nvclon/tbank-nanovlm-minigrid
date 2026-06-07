#!/usr/bin/env bash
set -euo pipefail

if [ ! -d "nanoVLM/models" ]; then
  echo "NanoVLM is missing."
  echo "Run: git clone https://github.com/huggingface/nanoVLM.git nanoVLM"
  exit 1
fi

echo "=== 1/8 Collecting expert dataset ==="
bash scripts/01_collect_data.sh

echo "=== 2/8 Training SFT-action ==="
bash scripts/02_train_sft_action.sh

echo "=== 3/8 Training SFT-text+action ==="
bash scripts/03_train_sft_text_action.sh

echo "=== 4/8 Training GRPO-action ==="
bash scripts/04_train_grpo_action.sh

echo "=== 5/8 Training GRPO-text+action ==="
bash scripts/05_train_grpo_text_action.sh

echo "=== 6/8 Running final evaluation ==="
bash scripts/06_eval_final.sh

echo "=== 7/8 Generating trajectory visualizations ==="
bash scripts/07_make_trajectories.sh

echo "=== 8/8 Exporting results and making figures ==="
bash scripts/09_export_results.sh

echo "Done. Final lightweight results are in results/"
