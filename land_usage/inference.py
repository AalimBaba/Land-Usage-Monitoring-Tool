from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .config import CLASS_NAMES, MODEL_INPUT_SIZE, NUM_CLASSES, PALETTE


class ModelUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class PredictionResult:
    mask: np.ndarray
    probabilities: np.ndarray
    mask_image: Image.Image
    overlay_image: Image.Image
    class_distribution: dict[str, float]
    mean_confidence: float


@lru_cache(maxsize=1)
def load_model(model_path: Path):
    if not model_path.exists():
        raise ModelUnavailableError(f"Model file not found at {model_path}. Train a model or add the bundled unet_model.h5.")
    try:
        import tensorflow as tf

        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as exc:
        raise ModelUnavailableError(f"Unable to load TensorFlow model: {exc}") from exc


def preprocess_image(image: Image.Image, size: tuple[int, int] = MODEL_INPUT_SIZE) -> np.ndarray:
    resized = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def colorize_mask(mask: np.ndarray) -> Image.Image:
    colors = np.asarray(PALETTE, dtype=np.uint8)
    return Image.fromarray(colors[np.clip(mask, 0, NUM_CLASSES - 1)])


def blend_overlay(image: Image.Image, mask_image: Image.Image) -> Image.Image:
    base = image.convert("RGB").resize(mask_image.size, Image.Resampling.BILINEAR)
    return Image.blend(base, mask_image.convert("RGB"), alpha=0.42)


def class_distribution(mask: np.ndarray) -> dict[str, float]:
    counts = np.bincount(mask.reshape(-1), minlength=NUM_CLASSES)
    total = counts.sum()
    if total == 0:
        return {name: 0.0 for name in CLASS_NAMES}
    return {CLASS_NAMES[idx]: float(counts[idx] * 100 / total) for idx in range(NUM_CLASSES)}


def predict_image(model, image: Image.Image) -> PredictionResult:
    batch = preprocess_image(image)
    probabilities = model.predict(batch, verbose=0)[0]
    mask = np.argmax(probabilities, axis=-1).astype(np.uint8)
    confidence = np.max(probabilities, axis=-1)
    mask_image = colorize_mask(mask)
    return PredictionResult(
        mask=mask,
        probabilities=probabilities,
        mask_image=mask_image,
        overlay_image=blend_overlay(image, mask_image),
        class_distribution=class_distribution(mask),
        mean_confidence=float(confidence.mean() * 100),
    )
