# NanoVLM MiniGrid Fine-tuning

This repository contains a reproducible pipeline for fine-tuning **NanoVLM** to control an agent in `MiniGrid-Empty-8x8-v0`.

The project compares:

- **SFT** on expert trajectories;
- **GRPO-style** RL fine-tuning;
- two output formats:
  - direct action: `left`, `right`, `forward`;
  - text + action: short explanation followed by `Action: ...`.

The final metrics, plots, and trajectory visualizations are included in `results/`.

---

## Quick start

Clone the repository:

```powershell
git clone https://github.com/nvclon/tbank-nanovlm-minigrid.git
cd tbank-nanovlm-minigrid
```

Create an environment:

```powershell
conda create -n nanovlm_minigrid python=3.10 -y
conda activate nanovlm_minigrid
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Clone NanoVLM into the project folder:

```powershell
git clone https://github.com/huggingface/nanoVLM.git nanoVLM
```

Run a quick smoke test on Windows:

```powershell
.\scripts\run_smoke_pipeline.ps1
```

Run a quick smoke test on Linux/macOS:

```bash
chmod +x scripts/*.sh
bash scripts/run_smoke_pipeline.sh
```

The smoke test checks that dataset generation, SFT, GRPO, evaluation, and trajectory rendering run end-to-end. It uses a tiny dataset and very few training steps, so its metrics are not expected to match the final report.

---

## Full reproduction

To reproduce the full experiment on Windows:

```powershell
.\scripts\run_full_pipeline.ps1
```

To reproduce the full experiment on Linux/macOS:

```bash
chmod +x scripts/*.sh
bash scripts/run_full_pipeline.sh
```

The full run performs:

1. expert dataset collection;
2. SFT action-only training;
3. SFT text+action training;
4. GRPO-action training;
5. GRPO-text+action training;
6. final evaluation;
7. trajectory visualization;
8. export of metrics, plots, and images to `results/`.

The full run can take several hours on a 6GB GPU.

After the run, the important lightweight artifacts are written to:

```text
results/
  metrics/
  figures/
  trajectories/
```

Large generated files are stored in:

```text
outputs/
data/
```

These folders are ignored by Git.

---

## Repository structure

```text
src/
  collect_expert_dataset.py
  train_sft.py
  train_grpo_action_fast.py
  train_grpo_text_action_free.py
  eval_zero_shot.py

  data_utils/
    formats.py
    sft_dataset.py

  expert/
    shortest_path_expert.py

  models/
    action_parser.py
    action_sampler.py
    nanovlm_policy.py

  eval/
    evaluate.py
    trajectory_examples.py
    qualitative_examples.py
    make_result_figures.py

scripts/
  run_smoke_pipeline.ps1
  run_full_pipeline.ps1
  run_smoke_pipeline.sh
  run_full_pipeline.sh
  01_collect_data.ps1
  02_train_sft_action.ps1
  03_train_sft_text_action.ps1
  04_train_grpo_action.ps1
  05_train_grpo_text_action.ps1
  06_eval_final.ps1
  07_make_trajectories.ps1
  08_make_figures.ps1
  09_export_results.ps1

results/
  metrics/
  figures/
  trajectories/

tools/
  debug/
  experiments/
```

---

## Dataset and expert

The expert is a shortest-path oracle for `MiniGrid-Empty-8x8-v0`.

It uses privileged simulator state only for labeling:

- agent position;
- agent direction;
- goal position.

The NanoVLM policy itself receives only rendered RGB images.

This expert is suitable for `EmptyEnv`, because the room has no internal obstacles. Therefore, the optimal next action can be computed directly and reproducibly.

To generate the dataset manually on Windows:

```powershell
.\scripts\01_collect_data.ps1
```

On Linux/macOS:

```bash
bash scripts/01_collect_data.sh
```

This creates:

```text
data/expert/
  train.jsonl
  val.jsonl
  test.jsonl
  summary.json
  images/
```

---

## Output formats

### Action-only

The model outputs exactly one action word:

```text
forward
```

Valid actions are:

```text
left
right
forward
```

### Text + action

The model outputs a short observation, a short plan, and the final action:

```text
Observation: The goal is not directly in front of the agent.
Plan: Turn right to face a better direction.
Action: right
```

During evaluation, the parser extracts the final `Action: ...` field.

---

## Training

### SFT action-only

Windows:

```powershell
.\scripts\02_train_sft_action.ps1
```

Linux/macOS:

```bash
bash scripts/02_train_sft_action.sh
```

The final reported SFT-action metrics use:

```text
outputs/main/sft_action/step_1000
```

This checkpoint gave the best closed-loop result among saved SFT-action checkpoints.

### SFT text+action

Windows:

```powershell
.\scripts\03_train_sft_text_action.ps1
```

Linux/macOS:

```bash
bash scripts/03_train_sft_text_action.sh
```

The best text+action setting freezes the vision encoder and trains:

- modality projector;
- last language decoder block.

This worked better than training only the last decoder block from the base model.

### GRPO-action

Windows:

```powershell
.\scripts\04_train_grpo_action.ps1
```

Linux/macOS:

```bash
bash scripts/04_train_grpo_action.sh
```

For tractability on a 6GB GPU, this is implemented as a one-step GRPO-style variant:

1. sample a group of candidate first actions;
2. estimate each action reward by rollout;
3. compute group-normalized advantages;
4. update the policy using the log-probability of the sampled action.

### GRPO text+action

Windows:

```powershell
.\scripts\05_train_grpo_text_action.ps1
```

Linux/macOS:

```bash
bash scripts/05_train_grpo_text_action.sh
```

This starts from the SFT text+action checkpoint and uses free-form generation. In the final run, generated samples inside each GRPO group usually mapped to the same parsed action and received the same reward. As a result, the GRPO updates were skipped because the group reward variance was zero.

---

## Evaluation

Run final evaluation.

Windows:

```powershell
.\scripts\06_eval_final.ps1
```

Linux/macOS:

```bash
bash scripts/06_eval_final.sh
```

Generate trajectory visualizations.

Windows:

```powershell
.\scripts\07_make_trajectories.ps1
```

Linux/macOS:

```bash
bash scripts/07_make_trajectories.sh
```

Export metrics, trajectories, and plots to `results/`.

Windows:

```powershell
.\scripts\09_export_results.ps1
```

Linux/macOS:

```bash
bash scripts/09_export_results.sh
```

Generate figures only.

Windows:

```powershell
.\scripts\08_make_figures.ps1
```

Linux/macOS:

```bash
bash scripts/08_make_figures.sh
```

---

## Results

Final evaluation uses 20 episodes with `max_episode_steps=30`.

| Method | Success rate | Avg return | Avg steps | Invalid rate | Notes |
|---|---:|---:|---:|---:|---|
| Zero-shot NanoVLM | 0.00 | 0.0000 | 30.00 | 0.00 | Collapses mostly to one action |
| Forward baseline | 0.10 | 0.0984 | 27.45 | 0.00 | Always moves forward |
| Expert | 1.00 | 0.9787 | 6.05 | 0.00 | Shortest-path oracle |
| SFT-action | 1.00 | 0.9773 | 6.45 | 0.00 | Selected checkpoint: `step_1000` |
| GRPO-action | 1.00 | 0.9787 | 6.05 | 0.00 | Small improvement over SFT-action |
| SFT-text+action | 1.00 | 0.9787 | 6.05 | 0.00 | Best text+action SFT checkpoint |
| GRPO-text+action | 1.00 | 0.9787 | 6.05 | 0.00 | No effective update |

The exact JSON metrics are in:

```text
results/metrics/
```

Plots are in:

```text
results/figures/
```

Trajectory visualizations are in:

```text
results/trajectories/
```

---

## Main observations

- Zero-shot NanoVLM can output valid action words, but it does not reliably control the agent.
- SFT is enough to solve `EmptyEnv`, because the task is simple and fully observable.
- GRPO-action slightly improves average episode length over SFT-action.
- Text+action SFT works well after longer training with the modality projector and the last decoder block trainable.
- GRPO-text+action does not improve over SFT-text+action in this setup because free-form generations become too deterministic.

---

## Notes

The repository does not include model checkpoints, generated datasets, or the NanoVLM source tree.

To reproduce everything from scratch on Windows:

```powershell
git clone https://github.com/nvclon/tbank-nanovlm-minigrid.git
cd tbank-nanovlm-minigrid
conda create -n nanovlm_minigrid python=3.10 -y
conda activate nanovlm_minigrid
pip install -r requirements.txt
git clone https://github.com/huggingface/nanoVLM.git nanoVLM
.\scripts\run_full_pipeline.ps1
```

For a quick check on Windows:

```powershell
.\scripts\run_smoke_pipeline.ps1
```

To reproduce everything from scratch on Linux/macOS:

```bash
git clone https://github.com/nvclon/tbank-nanovlm-minigrid.git
cd tbank-nanovlm-minigrid
conda create -n nanovlm_minigrid python=3.10 -y
conda activate nanovlm_minigrid
pip install -r requirements.txt
git clone https://github.com/huggingface/nanoVLM.git nanoVLM
chmod +x scripts/*.sh
bash scripts/run_full_pipeline.sh
```

For a quick check on Linux/macOS:

```bash
bash scripts/run_smoke_pipeline.sh
```
