mkdir outputs/final_rerun/eval_20

python -m src.eval.evaluate --policy expert --num-episodes 20 --max-episode-steps 30 --output-path outputs/final_rerun/eval_20/expert_eval_20.json

python -m src.eval.evaluate --policy forward --num-episodes 20 --max-episode-steps 30 --output-path outputs/final_rerun/eval_20/forward_eval_20.json

python -m src.eval.evaluate --policy nanovlm --checkpoint-path outputs/final_rerun/sft_action/step_1000 --output-format action_only --num-episodes 20 --max-episode-steps 30 --output-path outputs/final_rerun/eval_20/sft_action_eval_20.json

python -m src.eval.evaluate --policy nanovlm --checkpoint-path outputs/final_rerun/grpo_action/final --output-format action_only --num-episodes 20 --max-episode-steps 30 --output-path outputs/final_rerun/eval_20/grpo_action_eval_20.json

python -m src.eval.evaluate --policy nanovlm --checkpoint-path outputs/final_rerun/sft_text_action/final --output-format text_action --max-new-tokens 40 --num-episodes 20 --max-episode-steps 30 --output-path outputs/final_rerun/eval_20/sft_text_action_eval_20.json

python -m src.eval.evaluate --policy nanovlm --checkpoint-path outputs/final_rerun/grpo_text_action/final --output-format text_action --max-new-tokens 40 --num-episodes 20 --max-episode-steps 30 --output-path outputs/final_rerun/eval_20/grpo_text_action_eval_20.json