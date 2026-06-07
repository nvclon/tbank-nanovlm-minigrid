python -m src.eval.trajectory_examples --policy expert --output-format action_only --start-seed 90000 --num-episodes 4 --max-episode-steps 15 --output-dir outputs/final_rerun/trajectories/expert

python -m src.eval.trajectory_examples --checkpoint-path outputs/final_rerun/sft_action/step_1000 --output-format action_only --start-seed 90000 --num-episodes 4 --max-episode-steps 15 --max-new-tokens 8 --output-dir outputs/final_rerun/trajectories/sft_action

python -m src.eval.trajectory_examples --checkpoint-path outputs/final_rerun/grpo_action/final --output-format action_only --start-seed 90000 --num-episodes 4 --max-episode-steps 15 --max-new-tokens 8 --output-dir outputs/final_rerun/trajectories/grpo_action

python -m src.eval.trajectory_examples --checkpoint-path outputs/final_rerun/sft_text_action/final --output-format text_action --start-seed 90000 --num-episodes 4 --max-episode-steps 15 --max-new-tokens 40 --output-dir outputs/final_rerun/trajectories/sft_text_action

python -m src.eval.trajectory_examples --checkpoint-path outputs/final_rerun/grpo_text_action/final --output-format text_action --start-seed 90000 --num-episodes 4 --max-episode-steps 15 --max-new-tokens 40 --output-dir outputs/final_rerun/trajectories/grpo_text_action