#!/usr/bin/env bash
set -euo pipefail

if [ ! -d "nanoVLM/models" ]; then
  echo "NanoVLM is missing."
  echo "Run: git clone https://github.com/huggingface/nanoVLM.git nanoVLM"
  exit 1
fi

RUN_DIR="outputs/smoke"

echo "=== 1/7 Collecting tiny expert dataset ==="
python -m src.collect_expert_dataset \
  --train-episodes 80 \
  --val-episodes 20 \
  --test-episodes 20 \
  --output-dir data/expert_smoke

echo "=== 2/7 Training tiny SFT-action ==="
python -m src.train_sft \
  --train-jsonl data/expert_smoke/train.jsonl \
  --val-jsonl data/expert_smoke/val.jsonl \
  --output-format action_only \
  --train-mode mp \
  --max-steps 10 \
  --eval-every 5 \
  --save-every 10 \
  --eval-batches 2 \
  --max-length 512 \
  --lr 1e-4 \
  --output-dir "$RUN_DIR/sft_action"

echo "=== 3/7 Training tiny SFT-text+action ==="
python -m src.train_sft \
  --train-jsonl data/expert_smoke/train.jsonl \
  --val-jsonl data/expert_smoke/val.jsonl \
  --output-format text_action \
  --train-mode decoder_last \
  --max-steps 10 \
  --eval-every 5 \
  --save-every 10 \
  --eval-batches 2 \
  --max-length 512 \
  --lr 1e-5 \
  --output-dir "$RUN_DIR/sft_text_action"

echo "=== 4/7 Tiny GRPO-action ==="
python -m src.train_grpo_action_fast \
  --checkpoint-path "$RUN_DIR/sft_action/final" \
  --updates 2 \
  --group-size 2 \
  --eval-every 2 \
  --eval-episodes 2 \
  --save-every 2 \
  --temperature 1.2 \
  --lr 1e-6 \
  --max-episode-steps 10 \
  --output-dir "$RUN_DIR/grpo_action"

echo "=== 5/7 Tiny GRPO-text+action ==="
python -m src.train_grpo_text_action_free \
  --checkpoint-path "$RUN_DIR/sft_text_action/final" \
  --updates 2 \
  --group-size 2 \
  --temperature 2.0 \
  --eval-every 2 \
  --eval-episodes 2 \
  --max-new-tokens 30 \
  --max-episode-steps 10 \
  --lr 1e-6 \
  --output-dir "$RUN_DIR/grpo_text_action"

echo "=== 6/7 Tiny eval ==="
mkdir -p "$RUN_DIR/eval"

python -m src.eval.evaluate \
  --policy expert \
  --num-episodes 2 \
  --max-episode-steps 10 \
  --output-path "$RUN_DIR/eval/expert_eval.json"

python -m src.eval.evaluate \
  --policy nanovlm \
  --checkpoint-path "$RUN_DIR/sft_action/final" \
  --output-format action_only \
  --num-episodes 2 \
  --max-episode-steps 10 \
  --output-path "$RUN_DIR/eval/sft_action_eval.json"

python -m src.eval.evaluate \
  --policy nanovlm \
  --checkpoint-path "$RUN_DIR/sft_text_action/final" \
  --output-format text_action \
  --max-new-tokens 30 \
  --num-episodes 2 \
  --max-episode-steps 10 \
  --output-path "$RUN_DIR/eval/sft_text_action_eval.json"

echo "=== 7/7 Tiny trajectories ==="

python -m src.eval.trajectory_examples \
  --policy expert \
  --output-format action_only \
  --start-seed 90000 \
  --num-episodes 1 \
  --max-episode-steps 8 \
  --output-dir "$RUN_DIR/trajectories/expert"

python -m src.eval.trajectory_examples \
  --checkpoint-path "$RUN_DIR/sft_action/final" \
  --output-format action_only \
  --start-seed 90000 \
  --num-episodes 1 \
  --max-episode-steps 8 \
  --max-new-tokens 8 \
  --output-dir "$RUN_DIR/trajectories/sft_action"

echo "Smoke pipeline finished."
echo "Outputs are in $RUN_DIR"
