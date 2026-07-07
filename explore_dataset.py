from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from land_usage.data import discover_dataset, load_image, load_mask
from land_usage.inference import colorize_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect dataset layout and one aligned image/mask pair.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--output", default=Path("sample_image_mask.png"), type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train, val, test = discover_dataset(args.data_dir, seed=args.seed)
    print(f"train pairs: {len(train)}")
    print(f"validation pairs: {len(val)}")
    print(f"test pairs: {len(test)}")

    sample = train[0]
    image = load_image(sample.image_path)
    mask = load_mask(sample.mask_path)
    print("sample image:", sample.image_path)
    print("sample mask:", sample.mask_path)
    print("image shape:", image.shape)
    print("mask shape:", mask.shape)
    print("mask classes:", np.unique(mask).tolist())

    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(image)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(colorize_mask(mask))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
