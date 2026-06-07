#!/usr/bin/env bash
set -euo pipefail

python -m src.collect_expert_dataset \
  --train-episodes 3000 \
  --val-episodes 300 \
  --test-episodes 300
