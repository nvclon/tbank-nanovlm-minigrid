import sys
from functools import partial
from pathlib import Path

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NANOVLM_ROOT = PROJECT_ROOT / "nanoVLM"

sys.path.insert(0, str(NANOVLM_ROOT))

from data.processors import get_image_processor, get_tokenizer
from models.config import VLMConfig
from src.data_utils.sft_dataset import MiniGridSFTDataset, collate_sft_batch


def main():
    vlm_cfg = VLMConfig()
    vlm_cfg.vit_img_size = 512
    vlm_cfg.max_img_size = 512
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
        batch_size=2,
        shuffle=False,
        collate_fn=partial(
            collate_sft_batch,
            pad_token_id=tokenizer.pad_token_id,
        ),
    )

    batch = next(iter(loader))

    print("dataset size:", len(dataset))
    print("input_ids:", batch["input_ids"].shape)
    print("attention_mask:", batch["attention_mask"].shape)
    print("labels:", batch["labels"].shape)
    print("images len:", len(batch["images"]))
    print("image 0 shape:", batch["images"][0].shape)
    print("targets:", batch["targets"])
    print("action_names:", batch["action_names"])

    num_loss_tokens = (batch["labels"] != -100).sum().item()
    print("loss tokens:", num_loss_tokens)
    print("ok")


if __name__ == "__main__":
    main()