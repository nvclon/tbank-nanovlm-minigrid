import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NANOVLM_ROOT = PROJECT_ROOT / "nanoVLM"

sys.path.insert(0, str(NANOVLM_ROOT))

from models.config import VLMConfig, TrainConfig
from data.processors import get_image_processor, get_tokenizer


def main():
    vlm_cfg = VLMConfig()
    train_cfg = TrainConfig()

    print("VLM config:")
    print(vlm_cfg)

    print()
    print("Train config:")
    print(train_cfg)

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

    print()
    print("tokenizer:", type(tokenizer))
    print("tokenizer vocab size:", len(tokenizer))
    print("image_processor:", image_processor)
    print("ok")


if __name__ == "__main__":
    main()