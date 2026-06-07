import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NANOVLM_ROOT = PROJECT_ROOT / "nanoVLM"

sys.path.insert(0, str(NANOVLM_ROOT))

from data.processors import get_image_processor, get_tokenizer
from models.config import VLMConfig
from models.vision_language_model import VisionLanguageModel
from src.data_utils.sft_dataset import MiniGridSFTDataset, collate_sft_batch


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    vlm_cfg = VLMConfig()
    vlm_cfg.vit_img_size = 224
    vlm_cfg.max_img_size = 224
    vlm_cfg.lm_max_length = 512

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

    dataset = MiniGridSFTDataset(
        jsonl_path="data/expert/train.jsonl",
        tokenizer=tokenizer,
        image_processor=image_processor,
        vlm_cfg=vlm_cfg,
        output_format="action_only",
        max_length=vlm_cfg.lm_max_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=partial(
            collate_sft_batch,
            pad_token_id=tokenizer.pad_token_id,
        ),
    )

    batch = next(iter(loader))

    model = VisionLanguageModel(vlm_cfg, load_backbone=True).to(device)
    model.eval()

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    images = batch["images"]

    with torch.no_grad():
        logits, loss = model(
            input_ids=input_ids,
            images=images,
            attention_mask=attention_mask,
            targets=labels,
        )

    print("input_ids:", input_ids.shape)
    print("attention_mask:", attention_mask.shape)
    print("labels:", labels.shape)
    print("images len:", len(images))
    print("image 0:", images[0].shape)
    print("target:", batch["targets"])
    print("logits:", logits.shape)
    print("loss:", float(loss))
    print("ok")


if __name__ == "__main__":
    main()