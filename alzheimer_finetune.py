"""alzheimer_finetune.py

Utility script to convert an Alzheimer–MRI image dataset organised either as
<dataset_root>/<split>/<class_name>/img.jpg  *or*  <dataset_root>/<class_name>/img.jpg
into an OpenAI **vision fine‑tuning** JSONL file.

New in this revision
--------------------
* **--max-per-class** (int, optional) — cap the number of images taken from
  each class; handy for smoke‑testing the JSONL format with a tiny subset.

Example
-------
python alzheimer_finetune.py \
    --dataset-root /data/Alzheimer_MRI \
    --output-jsonl train.jsonl \
    --max-per-class 10

References
~~~~~~~~~~
* https://platform.openai.com/docs/guides/fine-tuning
* https://openai.com/index/introducing-vision-to-the-fine-tuning-api/
"""

from __future__ import annotations

import argparse
import base64
import json
import random
from io import BytesIO
from pathlib import Path

from PIL import Image

SYSTEM_PROMPT = (
    "You are a medical‑imaging assistant. Examine the provided brain MRI scan "
    "and answer with one of the following classes ONLY, exactly as written: "
    "{class_list}."
)


def encode_image(image_path: Path, quality: int = 95) -> str:
    """Read *image_path*, convert to RGB JPEG, return Base‑64 string."""
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode()


def build_example(encoded_img: str, label: str, class_list: list[str]) -> dict:
    """Return one training example in OpenAI chat‑completion format."""
    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(class_list=", ".join(class_list)),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "MRI scan of the brain. What is the diagnosis?",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_img}"
                        },
                    },
                ],
            },
            {"role": "assistant", "content": label + "\nNOTE: this is made by an AI model and may be incorrect. Always consult a medical professional for accurate diagnosis."},
        ]
    }


def gather_images(root: Path) -> list[Path]:
    """Return list of all jpg/jpeg/png images under *root*."""
    return sorted(root.glob("**/*.jpg")) + sorted(root.glob("**/*.jpeg")) + sorted(
        root.glob("**/*.png")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-jsonl", default="train.jsonl", type=Path)
    parser.add_argument("--val-jsonl", type=Path, default=None)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality 1–100")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Limit the number of images per class (useful for quick tests).",
    )
    args = parser.parse_args()

    image_paths = gather_images(args.dataset_root)
    if not image_paths:
        raise SystemExit("No images found — check --dataset-root path.")

    # Group images by class label inferred from immediate parent directory
    label_to_paths: dict[str, list[Path]] = {}
    for p in image_paths:
        label = p.parent.name
        label_to_paths.setdefault(label, []).append(p)

    class_list = sorted(label_to_paths)

    # Optionally cap per‑class count (after shuffling for randomness)
    if args.max_per_class is not None:
        for label, paths in label_to_paths.items():
            random.shuffle(paths)
            label_to_paths[label] = paths[: args.max_per_class]

    all_examples: list[dict] = []
    for label, paths in label_to_paths.items():
        for img_path in paths:
            encoded = encode_image(img_path, quality=args.quality)
            all_examples.append(build_example(encoded, label, class_list))

    random.shuffle(all_examples)

    # Train/validation split
    if args.val_jsonl is not None:
        val_size = int(len(all_examples) * args.val_split)
        val_examples, train_examples = all_examples[:val_size], all_examples[val_size:]

        for file_path, data in [
            (args.output_jsonl, train_examples),
            (args.val_jsonl, val_examples),
        ]:
            with open(file_path, "w", encoding="utf-8") as f:
                for ex in data:
                    json.dump(ex, f, ensure_ascii=False)
                    f.write("\n")
        print(
            f"Wrote {len(train_examples)} training and {len(val_examples)} "
            f"validation examples to {args.output_jsonl} / {args.val_jsonl}."
        )
    else:
        with open(args.output_jsonl, "w", encoding="utf-8") as f:
            for ex in all_examples:
                json.dump(ex, f, ensure_ascii=False)
                f.write("\n")
        print(f"Wrote {len(all_examples)} examples to {args.output_jsonl}.")


if __name__ == "__main__":
    main()
