#!/usr/bin/env bash
set -euo pipefail

python -m src.train_sft \
  --output-format action_only \
  --train-mode mp \
  --max-steps 3000 \
  --eval-every 250 \
  --save-every 1000 \
  --eval-batches 50 \
  --max-length 512 \
  --lr 1e-4 \
  --output-dir outputs/main/sft_action
