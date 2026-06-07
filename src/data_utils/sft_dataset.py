import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NANOVLM_ROOT = PROJECT_ROOT / "nanoVLM"

sys.path.insert(0, str(NANOVLM_ROOT))

from data.processors import get_image_string


class MiniGridSFTDataset(Dataset):
    def __init__(
        self,
        jsonl_path,
        tokenizer,
        image_processor,
        vlm_cfg,
        output_format="action_only",
        max_length=2048,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.vlm_cfg = vlm_cfg
        self.output_format = output_format
        self.max_length = max_length

        with self.jsonl_path.open("r", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.rows)

    def _get_prompt_and_target(self, row):
        if self.output_format == "action_only":
            return row["prompt_action_only"], row["target_action_only"]

        if self.output_format == "text_action":
            return row["prompt_text_action"], row["target_text_action"]

        raise ValueError(f"Unknown output_format: {self.output_format}")

    def __getitem__(self, idx):
        row = self.rows[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        images_tensor, image_grid_size = self.image_processor(image)

        prompt, target = self._get_prompt_and_target(row)

        image_string = get_image_string(
            tokenizer=self.tokenizer,
            splitted_image_counts=[image_grid_size],
            mp_image_token_length=self.vlm_cfg.mp_image_token_length,
        )

        user_messages = [
            {
                "role": "user",
                "content": image_string + "\n" + prompt,
            }
        ]

        prefix_text = self.tokenizer.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        full_text = prefix_text + target + "<|im_end|>\n"

        prefix_ids = self.tokenizer(
            prefix_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )["input_ids"][0]

        encoded = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]

        labels = torch.full_like(input_ids, -100)

        prefix_len = len(prefix_ids)
        seq_len = len(input_ids)

        if prefix_len < seq_len:
            labels[prefix_len - 1 : seq_len - 1] = input_ids[prefix_len:seq_len]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": images_tensor,
            "target": target,
            "action_name": row["action_name"],
        }


def collate_sft_batch(batch, pad_token_id):
    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids = []
    attention_mask = []
    labels = []
    images = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len

        input_ids.append(
            torch.nn.functional.pad(
                item["input_ids"],
                (pad_len, 0),
                value=pad_token_id,
            )
        )

        attention_mask.append(
            torch.nn.functional.pad(
                item["attention_mask"],
                (pad_len, 0),
                value=0,
            )
        )

        labels.append(
            torch.nn.functional.pad(
                item["labels"],
                (pad_len, 0),
                value=-100,
            )
        )

        images.append(item["images"])

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
        "images": images,
        "targets": [item["target"] for item in batch],
        "action_names": [item["action_name"] for item in batch],
    }