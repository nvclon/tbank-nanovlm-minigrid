# NanoVLM MiniGrid Fine-tuning

This repository contains a small reproducible pipeline for adapting a vision-and-language model, NanoVLM, to control an agent in `MiniGrid-Empty-8x8-v0`.

The project compares:

- supervised fine-tuning (SFT) on expert trajectories;
- GRPO-style reinforcement learning fine-tuning;
- two output formats:
  - direct action: `left`, `right`, `forward`;
  - text + action: short state/plan description followed by `Action: ...`.

The environment is intentionally simple. `EmptyEnv` can be solved by a symbolic shortest-path policy, so the goal of this project is not to outperform the oracle expert, but to build and evaluate the full VLM control pipeline.

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
  01_collect_data.ps1
  02_train_sft_action.ps1
  03_train_sft_text_action.ps1
  04_train_grpo_action.ps1
  05_train_grpo_text_action.ps1
  06_eval_final.ps1
  07_make_trajectories.ps1
  08_make_figures.ps1

results/
  metrics/
  figures/
  trajectories/

tools/
  debug/
  experiments/
```

Large checkpoints and generated datasets are not committed. They can be reproduced with the scripts in `scripts/`.

## Setup

The code was tested with Python 3.10 and CUDA 11.8.

Install dependencies:

```powershell
pip install -r requirements.txt
```

Clone NanoVLM into the project directory:

```powershell
git clone https://github.com/huggingface/nanoVLM.git nanoVLM
```

Expected directory layout:

```text
tbank-nanovlm-minigrid/
  nanoVLM/
  src/
  scripts/
  results/
```

## Dataset and expert

The expert is a shortest-path oracle for `MiniGrid-Empty-8x8-v0`.

The expert uses privileged simulator state only for labeling:

- agent position;
- agent direction;
- goal position.

The NanoVLM policy receives only rendered RGB observations.

This choice is appropriate for `EmptyEnv`, because the room contains no internal obstacles. Therefore, the shortest path can be computed directly and reproducibly.

Generate expert trajectories:

```powershell
scripts/01_collect_data.ps1
```

The generated dataset contains train/validation/test splits with image observations and target actions.

## Output formats

### Action-only format

The model outputs exactly one action word:

```text
forward
```

Valid outputs are:

```text
left
right
forward
```

### Text + action format

The model outputs a short state description, a short plan, and the final action:

```text
Observation: The goal is not directly in front of the agent.
Plan: Turn right to face a better direction.
Action: right
```

The parser uses the final `Action:` field.

## Training

### SFT action-only

```powershell
scripts/02_train_sft_action.ps1
```

For the final reported SFT-action metrics, `outputs/final_rerun/sft_action/step_1000` is used, because it gave the best closed-loop evaluation among saved checkpoints.

### SFT text+action

```powershell
scripts/03_train_sft_text_action.ps1
```

The best text+action setting freezes the vision encoder and trains:

- modality projector;
- the last language decoder block.

This worked better than training only the last decoder block from the base model.

### GRPO action-only

```powershell
scripts/04_train_grpo_action.ps1
```

For tractability on a 6GB GPU, this is implemented as a one-step GRPO-style variant:

1. sample a group of candidate first actions;
2. estimate each action reward by rollout;
3. compute group-normalized advantages;
4. update the policy using the log-probability of the sampled action.

### GRPO text+action

```powershell
scripts/05_train_grpo_text_action.ps1
```

This variant starts from the SFT text+action checkpoint and uses free-form text generation. In the final run, generated samples inside each GRPO group collapsed to the same parsed action, giving zero reward variance. Therefore, the GRPO updates were skipped. This is kept as a failure mode.

## Evaluation

Run final evaluation:

```powershell
scripts/06_eval_final.ps1
```

Generate trajectory visualizations:

```powershell
scripts/07_make_trajectories.ps1
```

Generate figures and summary table:

```powershell
scripts/08_make_figures.ps1
```

Final metrics are stored in:

```text
results/metrics/
```

Figures are stored in:

```text
results/figures/
```

Trajectory visualizations are stored in:

```text
results/trajectories/
```

## Final results

Final evaluation uses 20 episodes with `max_episode_steps=30`.

| Method | Success rate | Avg return | Avg steps | Invalid rate | Notes |
|---|---:|---:|---:|---:|---|
| Zero-shot NanoVLM | 0.00 | 0.0000 | 30.00 | 0.00 | Collapses mostly to one valid action |
| Forward baseline | 0.10 | 0.0984 | 27.45 | 0.00 | Always moves forward |
| Expert | 1.00 | 0.9787 | 6.05 | 0.00 | Shortest-path oracle |
| SFT-action | 1.00 | 0.9773 | 6.45 | 0.00 | Selected checkpoint: `step_1000` |
| GRPO-action | 1.00 | 0.9787 | 6.05 | 0.00 | Small improvement over SFT-action |
| SFT-text+action | 1.00 | 0.9787 | 6.05 | 0.00 | Best text+action checkpoint |
| GRPO-text+action | 1.00 | 0.9787 | 6.05 | 0.00 | No effective update; zero group reward variance |

The exact JSON files are in `results/metrics/`.

## Main observations

1. Zero-shot NanoVLM can output valid action words, but it does not perform reliable visual control.
2. SFT is sufficient to solve `EmptyEnv` because the environment is simple and fully observable.
3. The action-only SFT policy reaches 100% success.
4. GRPO-action gives a small improvement in average return and episode length.
5. Text+action generation works well after longer SFT with the modality projector and last decoder block trainable.
6. Training only the last decoder block from the base model was not enough: it fixed the output format but collapsed mostly to `forward`.
7. GRPO-text+action did not produce effective updates because free-form generation had no group diversity.

## Failure modes

- Dataset action imbalance: `forward` dominates expert trajectories.
- Zero-shot prompt sensitivity: the base model tends to collapse to a constant action.
- Over-aggressive GRPO can shift action probabilities too far and degrade behavior.
- Free-form text+action GRPO can become deterministic, producing zero reward variance inside GRPO groups.

## Future work

- Add KL regularization against the SFT reference policy.
- Use entropy bonuses to improve GRPO exploration.
- Try balanced sampling for SFT to reduce action imbalance.
- Evaluate on larger maps such as `MiniGrid-Empty-16x16-v0`.
- Test harder MiniGrid tasks with obstacles, doors, or keys.
- Use LoRA instead of full last-block fine-tuning.
- Compare constrained decoding and free-form decoding for text+action.
- Report wall-clock efficiency in addition to environment-step efficiency.
- Try curriculum learning from smaller to larger maps.
- Add policy KL/action-distribution plots before and after GRPO.

## Notes

The submitted repository does not include model checkpoints. To reproduce them, run the scripts in order from `scripts/`.

The `results/` directory contains lightweight metrics, figures, and trajectory visualizations used in the report.