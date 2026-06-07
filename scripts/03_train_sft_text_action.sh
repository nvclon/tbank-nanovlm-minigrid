#!/usr/bin/env bash
set -euo pipefail

python -m src.train_sft \
  --output-format text_action \
  --train-mode decoder_last \
  --max-steps 4000 \
  --eval-every 250 \
  --save-every 1000 \
  --eval-batches 50 \
  --max-length 512 \
  --lr 1e-5 \
  --output-dir outputs/main/sft_text_action
