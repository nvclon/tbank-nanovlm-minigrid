$ErrorActionPreference = "Stop"

$RunDir = "outputs/main"
$MetricsDir = "results/metrics"
$FiguresDir = "results/figures"
$TrajectoriesDir = "results/trajectories"

New-Item -ItemType Directory -Force $MetricsDir | Out-Null
New-Item -ItemType Directory -Force $FiguresDir | Out-Null

Copy-Item "$RunDir/eval_20/*.json" $MetricsDir -Force

Copy-Item "$RunDir/sft_action/metrics.json" "$MetricsDir/sft_action_train_metrics.json" -Force
Copy-Item "$RunDir/sft_text_action/metrics.json" "$MetricsDir/sft_text_action_train_metrics.json" -Force
Copy-Item "$RunDir/grpo_action/metrics.json" "$MetricsDir/grpo_action_train_metrics.json" -Force
Copy-Item "$RunDir/grpo_text_action/metrics.json" "$MetricsDir/grpo_text_action_train_metrics.json" -Force

if (Test-Path $TrajectoriesDir) {
    Remove-Item $TrajectoriesDir -Recurse -Force
}

New-Item -ItemType Directory -Force $TrajectoriesDir | Out-Null

Copy-Item "$RunDir/trajectories/expert" "$TrajectoriesDir/expert" -Recurse -Force
Copy-Item "$RunDir/trajectories/sft_action" "$TrajectoriesDir/sft_action" -Recurse -Force
Copy-Item "$RunDir/trajectories/grpo_action" "$TrajectoriesDir/grpo_action" -Recurse -Force
Copy-Item "$RunDir/trajectories/sft_text_action" "$TrajectoriesDir/sft_text_action" -Recurse -Force
Copy-Item "$RunDir/trajectories/grpo_text_action" "$TrajectoriesDir/grpo_text_action" -Recurse -Force

python -m src.eval.make_result_figures

Write-Host "Exported final metrics, trajectories and figures to results/"