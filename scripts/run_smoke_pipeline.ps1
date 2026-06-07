$ErrorActionPreference = "Stop"

if (-not (Test-Path "nanoVLM/models")) {
    Write-Error "NanoVLM is missing. Run: git clone https://github.com/huggingface/nanoVLM.git nanoVLM"
}

$RunDir = "outputs/smoke"
$ResultsDir = "results_smoke"

Write-Host "=== 1/7 Collecting tiny expert dataset ==="
python -m src.collect_expert_dataset `
  --train-episodes 80 `
  --val-episodes 20 `
  --test-episodes 20 `
  --output-dir data/expert_smoke

Write-Host "=== 2/7 Training tiny SFT-action ==="
python -m src.train_sft `
  --train-jsonl data/expert_smoke/train.jsonl `
  --val-jsonl data/expert_smoke/val.jsonl `
  --output-format action_only `
  --train-mode mp `
  --max-steps 10 `
  --eval-every 5 `
  --save-every 10 `
  --eval-batches 2 `
  --max-length 512 `
  --lr 1e-4 `
  --output-dir $RunDir/sft_action

Write-Host "=== 3/7 Training tiny SFT-text+action ==="
python -m src.train_sft `
  --train-jsonl data/expert_smoke/train.jsonl `
  --val-jsonl data/expert_smoke/val.jsonl `
  --output-format text_action `
  --train-mode decoder_last `
  --max-steps 10 `
  --eval-every 5 `
  --save-every 10 `
  --eval-batches 2 `
  --max-length 512 `
  --lr 1e-5 `
  --output-dir $RunDir/sft_text_action

Write-Host "=== 4/7 Tiny GRPO-action ==="
python -m src.train_grpo_action_fast `
  --checkpoint-path $RunDir/sft_action/final `
  --updates 2 `
  --group-size 2 `
  --eval-every 2 `
  --eval-episodes 2 `
  --save-every 2 `
  --temperature 1.2 `
  --lr 1e-6 `
  --max-episode-steps 10 `
  --output-dir $RunDir/grpo_action

Write-Host "=== 5/7 Tiny GRPO-text+action ==="
python -m src.train_grpo_text_action_free `
  --checkpoint-path $RunDir/sft_text_action/final `
  --updates 2 `
  --group-size 2 `
  --temperature 2.0 `
  --eval-every 2 `
  --eval-episodes 2 `
  --max-new-tokens 30 `
  --max-episode-steps 10 `
  --lr 1e-6 `
  --output-dir $RunDir/grpo_text_action

Write-Host "=== 6/7 Tiny eval ==="
New-Item -ItemType Directory -Force $RunDir/eval | Out-Null

python -m src.eval.evaluate `
  --policy expert `
  --num-episodes 2 `
  --max-episode-steps 10 `
  --output-path $RunDir/eval/expert_eval.json

python -m src.eval.evaluate `
  --policy nanovlm `
  --checkpoint-path $RunDir/sft_action/final `
  --output-format action_only `
  --num-episodes 2 `
  --max-episode-steps 10 `
  --output-path $RunDir/eval/sft_action_eval.json

python -m src.eval.evaluate `
  --policy nanovlm `
  --checkpoint-path $RunDir/sft_text_action/final `
  --output-format text_action `
  --max-new-tokens 30 `
  --num-episodes 2 `
  --max-episode-steps 10 `
  --output-path $RunDir/eval/sft_text_action_eval.json

Write-Host "=== 7/7 Tiny trajectories ==="
python -m src.eval.trajectory_examples `
  --policy expert `
  --output-format action_only `
  --start-seed 90000 `
  --num-episodes 1 `
  --max-episode-steps 8 `
  --output-dir $RunDir/trajectories/expert

python -m src.eval.trajectory_examples `
  --checkpoint-path $RunDir/sft_action/final `
  --output-format action_only `
  --start-seed 90000 `
  --num-episodes 1 `
  --max-episode-steps 8 `
  --max-new-tokens 8 `
  --output-dir $RunDir/trajectories/sft_action

Write-Host "Smoke pipeline finished."
Write-Host "Outputs are in $RunDir"