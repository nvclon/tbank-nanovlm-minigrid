import json
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NANOVLM_ROOT = PROJECT_ROOT / "nanoVLM"

sys.path.insert(0, str(NANOVLM_ROOT))

from models.config import VLMConfig
from data.processors import get_image_processor, get_tokenizer


def load_first_row(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.loads(next(f))


def main():
    vlm_cfg = VLMConfig()

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

    text = (
        "<|im_start|>user\n"
        f"<|image|>\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{target}<|im_end|>"
    )

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    processed_images = image_processor(image)
    
    print("input_ids shape:", encoded["input_ids"].shape)
    print("attention_mask shape:", encoded["attention_mask"].shape)
    print()
    print("processed_images type:", type(processed_images))

    if isinstance(processed_images, dict):
        for k, v in processed_images.items():
            print(k, type(v), getattr(v, "shape", None))
    else:
        print("processed_images:", processed_images)
        print("shape:", getattr(processed_images, "shape", None))

    print("ok")


if __name__ == "__main__":
    main()