import json
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NANOVLM_ROOT = PROJECT_ROOT / "nanoVLM"

sys.path.insert(0, str(NANOVLM_ROOT))

from data.processors import get_image_processor, get_image_string, get_tokenizer
from models.config import VLMConfig
from models.vision_language_model import VisionLanguageModel


def load_first_row(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.loads(next(f))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    vlm_cfg = VLMConfig()
    vlm_cfg.lm_max_length = 2048

    tokenizer = get_tokenizer(
        vlm_cfg.lm_tokenizer,
        vlm_cfg.vlm_extra_tokens,
        vlm_cfg.lm_chat_template,
    )

    image_processor = get_image_processor(
        vlm_cfg.max_img_size,
        vlm_cfg.vit_img_size,
        vlm_cfg.resize_to_max_side_len,
    )

    row = load_first_row("data/expert/train.jsonl")

    image = Image.open(row["image_path"]).convert("RGB")
    prompt = row["prompt_action_only"]
    target = row["target_action_only"]

    images_tensor, image_grid_size = image_processor(image)

    image_string = get_image_string(
        tokenizer=tokenizer,
        splitted_image_counts=[image_grid_size],
        mp_image_token_length=vlm_cfg.mp_image_token_length,
    )

    messages = [
        {
            "role": "user",
            "content": image_string + "\n" + prompt,
        },
        {
            "role": "assistant",
            "content": target,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=vlm_cfg.lm_max_length,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    targets = input_ids.clone()

    images = [images_tensor]

    model = VisionLanguageModel(vlm_cfg, load_backbone=True).to(device)
    model.eval()

    with torch.no_grad():
        logits, loss = model(
            input_ids=input_ids,
            images=images,
            attention_mask=attention_mask,
            targets=targets,
        )

    print("image_grid_size:", image_grid_size)
    print("images_tensor shape:", images_tensor.shape)
    print("input_ids shape:", input_ids.shape)
    print("logits shape:", logits.shape)
    print("loss:", float(loss))
    print("ok")


if __name__ == "__main__":
    main()