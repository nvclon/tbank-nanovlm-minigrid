import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NANOVLM_ROOT = PROJECT_ROOT / "nanoVLM"

sys.path.insert(0, str(NANOVLM_ROOT))

from models.config import VLMConfig
from data.processors import get_image_processor, get_image_string, get_tokenizer
from models.vision_language_model import VisionLanguageModel
from src.data_utils.formats import ACTION_ONLY_PROMPT, TEXT_ACTION_PROMPT
from src.models.action_parser import parse_action_only, parse_text_action


class NanoVLMPolicy:
    def __init__(
        self,
        checkpoint_path,
        output_format="action_only",
        device=None,
        greedy=True,
        max_new_tokens=8,
        prompt_override=None,
    ):
        self.checkpoint_path = checkpoint_path
        self.output_format = output_format
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.greedy = greedy
        self.max_new_tokens = max_new_tokens
        self.prompt_override = prompt_override

        if checkpoint_path == "base":
            self.cfg = VLMConfig()
            self.cfg.lm_max_length = 512
            self.cfg.vit_img_size = 512
            self.cfg.max_img_size = 512

            self.model = VisionLanguageModel(
                self.cfg,
                load_backbone=True,
            ).to(self.device)
        else:
            self.model = VisionLanguageModel.from_pretrained(
                checkpoint_path
            ).to(self.device)

            self.cfg = self.model.cfg

        self.model.eval()

        self.tokenizer = get_tokenizer(
            self.cfg.lm_tokenizer,
            self.cfg.vlm_extra_tokens,
            self.cfg.lm_chat_template,
        )

        self.image_processor = get_image_processor(
            self.cfg.max_img_size,
            self.cfg.vit_img_size,
            self.cfg.resize_to_max_side_len,
        )

    def _get_prompt(self):
        if self.prompt_override is not None:
            return self.prompt_override
        if self.output_format == "action_only":
            return ACTION_ONLY_PROMPT

        if self.output_format == "text_action":
            return TEXT_ACTION_PROMPT

        raise ValueError(f"Unknown output_format: {self.output_format}")

    def _parse(self, text):
        if self.output_format == "action_only":
            return parse_action_only(text)

        if self.output_format == "text_action":
            return parse_text_action(text)

        raise ValueError(f"Unknown output_format: {self.output_format}")

    def act(self, env):
        img_array = env.render()
        image = Image.fromarray(img_array).convert("RGB")

        images_tensor, image_grid_size = self.image_processor(image)

        image_string = get_image_string(
            tokenizer=self.tokenizer,
            splitted_image_counts=[image_grid_size],
            mp_image_token_length=self.cfg.mp_image_token_length,
        )

        messages = [
            {
                "role": "user",
                "content": image_string + "\n" + self._get_prompt(),
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.lm_max_length,
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                images=[images_tensor],
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                greedy=self.greedy,
                temperature=0.5,
            )

        generated_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )

        action_id, invalid = self._parse(generated_text)

        return action_id, invalid

    def generate_text(self, env):
        img_array = env.render()
        image = Image.fromarray(img_array).convert("RGB")

        images_tensor, image_grid_size = self.image_processor(image)

        image_string = get_image_string(
            tokenizer=self.tokenizer,
            splitted_image_counts=[image_grid_size],
            mp_image_token_length=self.cfg.mp_image_token_length,
        )

        messages = [
            {
                "role": "user",
                "content": image_string + "\n" + self._get_prompt(),
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.lm_max_length,
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                images=[images_tensor],
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                greedy=self.greedy,
                temperature=0.5,
            )

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)