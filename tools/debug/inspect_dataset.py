import json
from pathlib import Path

from PIL import Image


def main():
    jsonl_path = Path("data/expert/train.jsonl")

    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            rows.append(json.loads(line))
            if i >= 4:
                break

    for i, row in enumerate(rows):
        print("=" * 80)
        print("sample:", i)
        print("seed:", row["seed"])
        print("step:", row["step"])
        print("agent_pos:", row["agent_pos"])
        print("goal_pos:", row["goal_pos"])
        print("agent_dir:", row["agent_dir"])
        print("action:", row["action_name"])
        print("image:", row["image_path"])
        print("target_action_only:", row["target_action_only"])
        print("target_text_action:")
        print(row["target_text_action"])

        img_path = Path(row["image_path"])
        img = Image.open(img_path)
        print("image_size:", img.size)


if __name__ == "__main__":
    main()