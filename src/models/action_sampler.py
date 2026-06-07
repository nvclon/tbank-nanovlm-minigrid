import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NANOVLM_ROOT = PROJECT_ROOT / "nanoVLM"

sys.path.insert(0, str(NANOVLM_ROOT))

from data.processors import get_image_string

ACTIONS = ["left", "right", "forward"]
ACTION_TO_ID = {
    "left": 0,
    "right": 1,
    "forward": 2,
}


def build_prompt(policy, env):
    img_array = env.render()
    image = Image.fromarray(img_array).convert("RGB")

    images_tensor, image_grid_size = policy.image_processor(image)

    image_string = get_image_string(
        tokenizer=policy.tokenizer,
        splitted_image_counts=[image_grid_size],
        mp_image_token_length=policy.cfg.mp_image_token_length,
    )

    messages = [
        {
            "role": "user",
            "content": image_string + "\n" + policy._get_prompt(),
        }
    ]

    prefix_text = policy.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return prefix_text, images_tensor


def score_action(policy, prefix_text, images_tensor, action, requires_grad=False):
    full_text = prefix_text + action + "<|im_end|>\n"

    prefix_ids = policy.tokenizer(
        prefix_text,
        return_tensors="pt",
        truncation=True,
        max_length=policy.cfg.lm_max_length,
    )["input_ids"][0]

    encoded = policy.tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=policy.cfg.lm_max_length,
    )

    input_ids = encoded["input_ids"].to(policy.device)
    attention_mask = encoded["attention_mask"].to(policy.device)

    labels = torch.full_like(input_ids, -100)

    prefix_len = len(prefix_ids)
    seq_len = input_ids.shape[1]

    if prefix_len < seq_len:
        labels[:, prefix_len - 1 : seq_len - 1] = input_ids[:, prefix_len:seq_len]

    def forward_score():
        logits, _ = policy.model(
            input_ids=input_ids,
            images=[images_tensor],
            attention_mask=attention_mask,
            targets=None,
        )

        logits = policy.model.decoder.head(logits)
        log_probs = torch.log_softmax(logits, dim=-1)

        mask = labels != -100
        token_log_probs = log_probs.gather(
            -1,
            labels.clamp_min(0).unsqueeze(-1),
        ).squeeze(-1)

        return token_log_probs[mask].sum()

    if requires_grad:
        return forward_score()

    with torch.no_grad():
        return forward_score()


def action_distribution(policy, env, temperature=1.0):
    prefix_text, images_tensor = build_prompt(policy, env)

    logprobs = []

    for action in ACTIONS:
        lp = score_action(policy, prefix_text, images_tensor, action)
        logprobs.append(lp)

    logprobs = torch.stack(logprobs)
    probs = torch.softmax(logprobs / temperature, dim=0)

    return {
        "actions": ACTIONS,
        "action_ids": [ACTION_TO_ID[a] for a in ACTIONS],
        "logprobs": logprobs.detach().cpu(),
        "probs": probs.detach().cpu(),
    }


def sample_action(policy, env, temperature=1.0):
    dist = action_distribution(policy, env, temperature=temperature)

    probs = dist["probs"]
    idx = torch.multinomial(probs, num_samples=1).item()

    return {
        "action": dist["actions"][idx],
        "action_id": dist["action_ids"][idx],
        "logprob": dist["logprobs"][idx],
        "prob": dist["probs"][idx],
        "dist": dist,
    }