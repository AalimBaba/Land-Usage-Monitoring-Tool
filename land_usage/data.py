from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .config import MODEL_INPUT_SIZE, NUM_CLASSES


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MASK_EXTENSIONS = {".tif", ".tiff", ".png"}


@dataclass(frozen=True)
class DatasetPair:
    image_path: Path
    mask_path: Path


def find_pairs(image_dir: Path, mask_dir: Path) -> list[DatasetPair]:
    images = {path.stem: path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
    masks = {path.stem: path for path in mask_dir.iterdir() if path.suffix.lower() in MASK_EXTENSIONS}
    keys = sorted(images.keys() & masks.keys())
    return [DatasetPair(images[key], masks[key]) for key in keys]


def resolve_split_dirs(data_dir: Path, split: str) -> tuple[Path, Path]:
    return data_dir / f"{split}_images", data_dir / f"{split}_masks"


def load_image(path: Path, size: tuple[int, int] = MODEL_INPUT_SIZE) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_mask(path: Path, size: tuple[int, int] = MODEL_INPUT_SIZE) -> np.ndarray:
    try:
        import rasterio

        with rasterio.open(path) as src:
            mask = src.read(1)
        mask_image = Image.fromarray(mask.astype(np.uint8))
    except Exception:
        mask_image = Image.open(path).convert("L")

    mask = np.asarray(mask_image.resize(size, Image.Resampling.NEAREST), dtype=np.int64)
    unique = np.unique(mask)
    if unique.size and unique.min() >= 1 and unique.max() <= NUM_CLASSES:
        mask = mask - 1
    return np.clip(mask, 0, NUM_CLASSES - 1)


def load_pairs(pairs: Iterable[DatasetPair], size: tuple[int, int] = MODEL_INPUT_SIZE) -> tuple[np.ndarray, np.ndarray]:
    images, masks = [], []
    for pair in pairs:
        images.append(load_image(pair.image_path, size))
        masks.append(load_mask(pair.mask_path, size))
    if not images:
        raise ValueError("No valid image/mask pairs were found.")
    return np.stack(images), np.stack(masks)


def split_pairs(
    pairs: list[DatasetPair],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[DatasetPair], list[DatasetPair], list[DatasetPair]]:
    if len(pairs) < 3:
        raise ValueError("At least three image/mask pairs are required for train/validation/test splitting.")
    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)
    test_count = max(1, round(len(shuffled) * test_fraction))
    val_count = max(1, round(len(shuffled) * val_fraction))
    test = shuffled[:test_count]
    val = shuffled[test_count : test_count + val_count]
    train = shuffled[test_count + val_count :]
    if not train:
        raise ValueError("Split settings left no training samples.")
    return train, val, test


def discover_dataset(data_dir: Path, seed: int = 42) -> tuple[list[DatasetPair], list[DatasetPair], list[DatasetPair]]:
    train_dirs = resolve_split_dirs(data_dir, "train")
    val_dirs = resolve_split_dirs(data_dir, "val")
    test_dirs = resolve_split_dirs(data_dir, "test")

    if all(path.exists() for path in (*train_dirs, *val_dirs, *test_dirs)):
        return (
            find_pairs(*train_dirs),
            find_pairs(*val_dirs),
            find_pairs(*test_dirs),
        )

    image_dir = data_dir / "images"
    mask_dir = data_dir / "masks"
    if image_dir.exists() and mask_dir.exists():
        return split_pairs(find_pairs(image_dir, mask_dir), seed=seed)

    raise FileNotFoundError(
        "Expected split folders train_images/train_masks/val_images/val_masks/test_images/test_masks "
        "or unsplit images/masks folders."
    )

