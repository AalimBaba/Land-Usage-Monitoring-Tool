from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image


RelevanceStatus = Literal["Suitable", "Uncertain", "Rejected"]


@dataclass(frozen=True)
class InputValidationResult:
    file_integrity: Literal["Passed"]
    image_relevance: RelevanceStatus
    segmentation: Literal["Run", "Not run"]
    reasons: tuple[str, ...]

    @property
    def is_suitable(self) -> bool:
        return self.image_relevance == "Suitable"

    @property
    def is_uncertain(self) -> bool:
        return self.image_relevance == "Uncertain"

    @property
    def is_rejected(self) -> bool:
        return self.image_relevance == "Rejected"


def _entropy(gray: np.ndarray) -> float:
    hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    probs = hist[hist > 0] / gray.size
    return float(-(probs * np.log2(probs)).sum())


def _edge_density(gray: np.ndarray) -> float:
    diff_x = np.abs(np.diff(gray.astype(np.int16), axis=1)) > 35
    diff_y = np.abs(np.diff(gray.astype(np.int16), axis=0)) > 35
    return float((diff_x.mean() + diff_y.mean()) / 2)


def _color_features(rgb: np.ndarray) -> dict[str, float]:
    arr = rgb.astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    delta = maxc - minc
    saturation = np.where(maxc > 0, delta / maxc, 0.0)

    gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
    green = (g > r * 1.05) & (g > b * 1.05) & (g > 55)
    water = (b > g * 1.05) & (b > r * 1.10) & (b > 55)
    soil = (r > 70) & (g > 45) & (b < 150) & (np.abs(r - g) < 90) & (r >= b * 0.95)
    urban_gray = (saturation < 0.24) & (gray > 65) & (gray < 215)
    skin = (r > 95) & (g > 40) & (b > 20) & (r > g) & (r > b) & ((r - g) > 15) & (delta > 15)

    h, w = gray.shape
    central_skin = skin[h // 5 : h * 4 // 5, w // 5 : w * 4 // 5].mean()

    return {
        "entropy": _entropy(gray),
        "std": float(gray.std()),
        "edge_density": _edge_density(gray),
        "light_ratio": float((gray > 220).mean()),
        "white_ratio": float(((r > 235) & (g > 235) & (b > 235)).mean()),
        "dark_ratio": float((gray < 80).mean()),
        "mean_saturation": float(saturation.mean()),
        "natural_ratio": float((green | water | soil | urban_gray).mean()),
        "skin_ratio": float(skin.mean()),
        "central_skin_ratio": float(central_skin),
        "unique_rounded_colors": float(np.unique((rgb // 32).reshape(-1, 3), axis=0).shape[0]),
    }


def validate_land_image(image: Image.Image) -> InputValidationResult:
    """Reject obvious non-land imagery before running the segmentation model.

    This is a lightweight semantic relevance gate designed for Streamlit Cloud.
    It uses document/text, blank-image, skin-tone, texture, and land-color cues.
    The gate is intentionally conservative and is not a calibrated classifier.
    """

    resized = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
    rgb = np.asarray(resized, dtype=np.uint8)
    features = _color_features(rgb)
    reasons: list[str] = []

    if features["std"] < 8 or features["entropy"] < 2.0:
        return InputValidationResult(
            file_integrity="Passed",
            image_relevance="Rejected",
            segmentation="Not run",
            reasons=("Image is blank or nearly blank.",),
        )

    text_heavy_document = (
        features["light_ratio"] > 0.55
        and features["dark_ratio"] > 0.01
        and features["mean_saturation"] < 0.25
        and features["edge_density"] > 0.015
    )
    sparse_document = features["white_ratio"] > 0.45 and features["mean_saturation"] < 0.20 and features["natural_ratio"] < 0.35
    if text_heavy_document or sparse_document:
        return InputValidationResult(
            file_integrity="Passed",
            image_relevance="Rejected",
            segmentation="Not run",
            reasons=("Image appears to be a document, certificate, screenshot, or text-heavy graphic.",),
        )

    if features["central_skin_ratio"] > 0.22 or (features["skin_ratio"] > 0.28 and features["natural_ratio"] < 0.75):
        return InputValidationResult(
            file_integrity="Passed",
            image_relevance="Rejected",
            segmentation="Not run",
            reasons=("Image appears to contain a person or portrait rather than aerial land imagery.",),
        )

    if features["unique_rounded_colors"] < 12 and features["entropy"] < 3.1 and features["edge_density"] < 0.05:
        return InputValidationResult(
            file_integrity="Passed",
            image_relevance="Rejected",
            segmentation="Not run",
            reasons=("Image appears to be a simple synthetic graphic rather than land imagery.",),
        )

    if features["natural_ratio"] >= 0.42 and features["entropy"] >= 3.0 and features["edge_density"] >= 0.005:
        return InputValidationResult(
            file_integrity="Passed",
            image_relevance="Suitable",
            segmentation="Run",
            reasons=("Image has land/aerial-like colour and texture patterns.",),
        )

    if features["natural_ratio"] >= 0.25 and features["entropy"] >= 2.7:
        reasons.append("Image has some land-like visual cues, but relevance is uncertain.")
        return InputValidationResult(
            file_integrity="Passed",
            image_relevance="Uncertain",
            segmentation="Not run",
            reasons=tuple(reasons),
        )

    return InputValidationResult(
        file_integrity="Passed",
        image_relevance="Rejected",
        segmentation="Not run",
        reasons=("Image does not appear to be satellite or aerial land imagery.",),
    )
