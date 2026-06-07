python -m src.eval.trajectory_examples --policy expert --output-format action_only --start-seed 90000 --num-episodes 4 --max-episode-steps 15 --output-dir outputs/main/trajectories/expert

python -m src.eval.trajectory_examples --checkpoint-path outputs/main/sft_action/step_1000 --output-format action_only --start-seed 90000 --num-episodes 4 --max-episode-steps 15 --max-new-tokens 8 --output-dir outputs/main/trajectories/sft_action

python -m src.eval.trajectory_examples --checkpoint-path outputs/main/grpo_action/final --output-format action_only --start-seed 90000 --num-episodes 4 --max-episode-steps 15 --max-new-tokens 8 --output-dir outputs/main/trajectories/grpo_action

python -m src.eval.trajectory_examples --checkpoint-path outputs/main/sft_text_action/final --output-format text_action --start-seed 90000 --num-episodes 4 --max-episode-steps 15 --max-new-tokens 40 --output-dir outputs/main/trajectories/sft_text_action

python -m src.eval.trajectory_examples --checkpoint-path outputs/main/grpo_text_action/final --output-format text_action --start-seed 90000 --num-episodes 4 --max-episode-steps 15 --max-new-tokens 40 --output-dir outputs/main/trajectories/grpo_text_action
