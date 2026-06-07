import argparse
import json
import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NANOVLM_ROOT = PROJECT_ROOT / "nanoVLM"

sys.path.insert(0, str(NANOVLM_ROOT))

from data.processors import get_image_processor, get_tokenizer
from models.config import VLMConfig
from models.vision_language_model import VisionLanguageModel
from src.data_utils.sft_dataset import MiniGridSFTDataset, collate_sft_batch


def set_trainable_params(model, mode):
    for p in model.parameters():
        p.requires_grad = False

    if mode == "mp":
        for p in model.MP.parameters():
            p.requires_grad = True

    elif mode == "decoder_last":
        for p in model.MP.parameters():
            p.requires_grad = True

        for block in model.decoder.blocks[-1:]:
            for p in block.parameters():
                p.requires_grad = True

    elif mode == "decoder_only_last":
        for block in model.decoder.blocks[-1:]:
            for p in block.parameters():
                p.requires_grad = True

    else:
        raise ValueError(f"Unknown train mode: {mode}")


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_eval_loss(model, loader, device, max_batches):
    model.eval()
    losses = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            _, loss = model(
                input_ids=input_ids,
                images=batch["images"],
                attention_mask=attention_mask,
                targets=labels,
            )

            losses.append(float(loss))

    model.train()

    return sum(losses) / max(len(losses), 1)


def save_metrics(path, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-jsonl", type=str, default="data/expert/train.jsonl")
    parser.add_argument("--val-jsonl", type=str, default="data/expert/val.jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs/sft_action")

    parser.add_argument("--output-format", type=str, default="action_only", choices=["action_only", "text_action"])
    parser.add_argument("--train-mode", type=str, default="mp", choices=["mp", "decoder_last", "decoder_only_last"])

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--vit-img-size", type=int, default=512)
    parser.add_argument("--max-img-size", type=int, default=512)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vlm_cfg = VLMConfig()
    vlm_cfg.lm_max_length = args.max_length
    vlm_cfg.vit_img_size = args.vit_img_size
    vlm_cfg.max_img_size = args.max_img_size
    
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

    train_dataset = MiniGridSFTDataset(
        jsonl_path=args.train_jsonl,
        tokenizer=tokenizer,
        image_processor=image_processor,
        vlm_cfg=vlm_cfg,
        output_format=args.output_format,
        max_length=args.max_length,
    )

    val_dataset = MiniGridSFTDataset(
        jsonl_path=args.val_jsonl,
        tokenizer=tokenizer,
        image_processor=image_processor,
        vlm_cfg=vlm_cfg,
        output_format=args.output_format,
        max_length=args.max_length,
    )

    collate_fn = partial(
        collate_sft_batch,
        pad_token_id=tokenizer.pad_token_id,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = VisionLanguageModel(vlm_cfg, load_backbone=True).to(device)
    set_trainable_params(model, args.train_mode)
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    trainable_params = count_trainable_params(model)
    print("device:", device)
    print("train examples:", len(train_dataset))
    print("val examples:", len(val_dataset))
    print("trainable params:", trainable_params)
    print("output dir:", output_dir)

    metrics = {
        "args": vars(args),
        "trainable_params": trainable_params,
        "steps": [],
    }

    global_step = 0
    running_loss = 0.0

    progress = tqdm(total=args.max_steps)

    while global_step < args.max_steps:
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            _, loss = model(
                input_ids=input_ids,
                images=batch["images"],
                attention_mask=attention_mask,
                targets=labels,
            )

            loss = loss / args.grad_accum_steps
            loss.backward()

            running_loss += float(loss) * args.grad_accum_steps

            if (global_step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    1.0,
                )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress.update(1)

            if global_step % args.eval_every == 0:
                val_loss = run_eval_loss(
                    model=model,
                    loader=val_loader,
                    device=device,
                    max_batches=args.eval_batches,
                )

                avg_train_loss = running_loss / args.eval_every
                running_loss = 0.0

                row = {
                    "step": global_step,
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                }

                metrics["steps"].append(row)
                save_metrics(output_dir / "metrics.json", metrics)

                print(row)

            if global_step % args.save_every == 0:
                ckpt_dir = output_dir / f"step_{global_step}"
                model.save_pretrained(str(ckpt_dir))

            if global_step >= args.max_steps:
                break

    progress.close()

    model.save_pretrained(str(output_dir / "final"))
    save_metrics(output_dir / "metrics.json", metrics)
    print("saved final checkpoint:", output_dir / "final")


if __name__ == "__main__":
    main()