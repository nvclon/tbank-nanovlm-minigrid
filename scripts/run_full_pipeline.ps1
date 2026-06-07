$ErrorActionPreference = "Stop"

if (-not (Test-Path "nanoVLM/models")) {
    Write-Error "NanoVLM is missing. Run: git clone https://github.com/huggingface/nanoVLM.git nanoVLM"
}

Write-Host "=== 1/8 Collecting expert dataset ==="
& .\scripts\01_collect_data.ps1

Write-Host "=== 2/8 Training SFT-action ==="
& .\scripts\02_train_sft_action.ps1

Write-Host "=== 3/8 Training SFT-text+action ==="
& .\scripts\03_train_sft_text_action.ps1

Write-Host "=== 4/8 Training GRPO-action ==="
& .\scripts\04_train_grpo_action.ps1

Write-Host "=== 5/8 Training GRPO-text+action ==="
& .\scripts\05_train_grpo_text_action.ps1

Write-Host "=== 6/8 Running final evaluation ==="
& .\scripts\06_eval_final.ps1

Write-Host "=== 7/8 Generating trajectory visualizations ==="
& .\scripts\07_make_trajectories.ps1

Write-Host "=== 8/8 Exporting results and making figures ==="
& .\scripts\09_export_results.ps1

Write-Host "Done. Final lightweight results are in results/"