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
    diagnostics: dict[str, float | int | str | list[str]]

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


def _edge_map(gray: np.ndarray, threshold: int = 35) -> np.ndarray:
    diff_x = np.zeros_like(gray, dtype=bool)
    diff_y = np.zeros_like(gray, dtype=bool)
    diff_x[:, 1:] = np.abs(np.diff(gray.astype(np.int16), axis=1)) > threshold
    diff_y[1:, :] = np.abs(np.diff(gray.astype(np.int16), axis=0)) > threshold
    return diff_x | diff_y


def _tile_stats(gray: np.ndarray, tile: int = 32) -> tuple[float, float, float, float]:
    h, w = gray.shape
    stds = []
    means = []
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            patch = gray[y : y + tile, x : x + tile]
            if patch.size:
                stds.append(float(patch.std()))
                means.append(float(patch.mean()))
    std_arr = np.asarray(stds, dtype=np.float32)
    mean_arr = np.asarray(means, dtype=np.float32)
    return float(std_arr.mean()), float((std_arr > 14).mean()), float(mean_arr.std()), float((std_arr < 8).mean())


def _extract_features(image: Image.Image) -> dict[str, float | int]:
    width, height = image.size
    resized = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
    rgb = np.asarray(resized, dtype=np.uint8)
    arr = rgb.astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    delta = maxc - minc
    saturation = np.divide(delta, maxc, out=np.zeros_like(delta), where=maxc > 0)

    gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
    top_band = slice(0, 38)
    top_third = slice(0, 86)
    bottom_third = slice(170, 256)
    top_dark_ratio = float((gray[top_band, :] < 75).mean())
    top_sky_ratio = float(((b[top_third, :] > 120) & (b[top_third, :] > r[top_third, :] * 1.12) & (b[top_third, :] > g[top_third, :] * 1.03)).mean())
    warm_indoor = ((r > 110) & (g > 85) & (b < 150) & (r >= b * 1.15) & (saturation < 0.55))
    neutral_surface = (saturation < 0.20) & (gray > 90) & (gray < 235)
    edges = _edge_map(gray)
    row_edge = edges.mean(axis=1)
    col_edge = edges.mean(axis=0)

    dark_text = (gray < 95) & (saturation < 0.65)
    row_dark = dark_text.mean(axis=1)
    col_dark = dark_text.mean(axis=0)
    text_row_ratio = float(((row_dark > 0.012) & (row_dark < 0.45)).mean())
    text_col_ratio = float(((col_dark > 0.012) & (col_dark < 0.55)).mean())
    horizontal_line_density = float((row_edge > 0.20).mean())
    vertical_line_density = float((col_edge > 0.20).mean())
    right_vertical_stripes = float((col_edge[int(col_edge.size * 0.65) :] > 0.18).mean())
    row_partition_strength = float(row_edge.max())
    col_partition_strength = float(col_edge.max())

    border = np.zeros_like(edges, dtype=bool)
    band = 12
    border[:band, :] = True
    border[-band:, :] = True
    border[:, :band] = True
    border[:, -band:] = True
    border_edge_density = float(edges[border].mean())
    inner_edge_density = float(edges[~border].mean())
    rectangular_boundary_score = border_edge_density / max(inner_edge_density, 0.001)

    green = (g > r * 1.05) & (g > b * 1.05) & (g > 55)
    water = (b > g * 1.05) & (b > r * 1.10) & (b > 55)
    soil = (r > 70) & (g > 45) & (b < 150) & (np.abs(r - g) < 90) & (r >= b * 0.95)
    urban_gray = (saturation < 0.28) & (gray > 55) & (gray < 220)
    natural_ratio = float((green | water | soil | urban_gray).mean())
    land_color_mix = sum(float(mask.mean()) > 0.05 for mask in (green, water, soil, urban_gray))

    skin = (r > 95) & (g > 40) & (b > 20) & (r > g) & (r > b) & ((r - g) > 15) & (delta > 15)
    h, w = gray.shape
    central_skin = float(skin[h // 5 : h * 4 // 5, w // 5 : w * 4 // 5].mean())
    left_photo_skin = float(skin[h // 5 : h * 4 // 5, : w // 2].mean())

    tile_texture_mean, distributed_detail_ratio, spatial_variance, smooth_tile_ratio = _tile_stats(gray)
    unique_rounded_colors = int(np.unique((rgb // 32).reshape(-1, 3), axis=0).shape[0])
    top_region_std = float(gray[top_third, :].std())
    middle_region_std = float(gray[86:170, :].std())
    bottom_region_std = float(gray[bottom_third, :].std())
    smooth_region_count = int(sum(value < 18 for value in (top_region_std, middle_region_std, bottom_region_std)))
    band_mean_gap = float(
        max(
            abs(float(gray[top_third, :].mean()) - float(gray[86:170, :].mean())),
            abs(float(gray[bottom_third, :].mean()) - float(gray[86:170, :].mean())),
            abs(float(gray[top_third, :].mean()) - float(gray[bottom_third, :].mean())),
        )
    )

    return {
        "width": int(width),
        "height": int(height),
        "aspect_ratio": float(width / max(height, 1)),
        "entropy": _entropy(gray),
        "edge_density": float(edges.mean()),
        "line_density": float((horizontal_line_density + vertical_line_density) / 2),
        "horizontal_line_density": horizontal_line_density,
        "vertical_line_density": vertical_line_density,
        "right_vertical_stripe_ratio": right_vertical_stripes,
        "row_partition_strength": row_partition_strength,
        "col_partition_strength": col_partition_strength,
        "rectangular_boundary_score": float(rectangular_boundary_score),
        "text_density_estimate": float((text_row_ratio + text_col_ratio) / 2),
        "text_row_ratio": text_row_ratio,
        "text_col_ratio": text_col_ratio,
        "colour_diversity": unique_rounded_colors,
        "mean_saturation": float(saturation.mean()),
        "low_saturation_ratio": float((saturation < 0.18).mean()),
        "light_ratio": float((gray > 220).mean()),
        "white_ratio": float(((r > 235) & (g > 235) & (b > 235)).mean()),
        "dark_ratio": float((gray < 80).mean()),
        "top_dark_ratio": top_dark_ratio,
        "top_sky_ratio": top_sky_ratio,
        "texture_mean": tile_texture_mean,
        "distributed_detail_ratio": distributed_detail_ratio,
        "smooth_tile_ratio": smooth_tile_ratio,
        "top_region_std": top_region_std,
        "middle_region_std": middle_region_std,
        "bottom_region_std": bottom_region_std,
        "smooth_region_count": smooth_region_count,
        "band_mean_gap": band_mean_gap,
        "spatial_variance": spatial_variance,
        "natural_ratio": natural_ratio,
        "land_color_mix": int(land_color_mix),
        "skin_ratio": float(skin.mean()),
        "central_skin_ratio": central_skin,
        "left_photo_skin_ratio": left_photo_skin,
        "warm_indoor_ratio": float(warm_indoor.mean()),
        "neutral_surface_ratio": float(neutral_surface.mean()),
    }


def _score_indoor(features: dict[str, float | int]) -> tuple[int, list[str]]:
    score = 0
    fired: list[str] = []

    if features["top_sky_ratio"] > 0.70 and features["edge_density"] < 0.025:
        score += 2
        fired.append("ground-level horizon/sky band")
    if (
        features["smooth_region_count"] >= 2
        and features["band_mean_gap"] > 25
        and features["land_color_mix"] <= 3
        and (features["neutral_surface_ratio"] > 0.12 or features["smooth_tile_ratio"] > 0.20)
    ):
        score += 2
        fired.append("ceiling/wall/floor-like smooth bands")
    if features["smooth_tile_ratio"] > 0.25 and features["spatial_variance"] > 24 and features["distributed_detail_ratio"] < 0.55:
        score += 2
        fired.append("large smooth perspective surfaces")
    if (
        features["row_partition_strength"] > 0.28
        and features["col_partition_strength"] > 0.22
        and features["colour_diversity"] < 150
        and features["neutral_surface_ratio"] > 0.12
    ):
        score += 1
        fired.append("room-like wall/window/furniture partitions")
    if features["warm_indoor_ratio"] > 0.35 and features["distributed_detail_ratio"] < 0.45:
        score += 2
        fired.append("large warm indoor/object-like surfaces")
    if features["neutral_surface_ratio"] > 0.45 and features["texture_mean"] < 24:
        score += 1
        fired.append("large smooth wall/floor-like surfaces")
    if features["smooth_tile_ratio"] > 0.30 and features["land_color_mix"] <= 2:
        score += 1
        fired.append("large smooth interior/object regions")
    if features["warm_indoor_ratio"] > 0.25 and features["neutral_surface_ratio"] > 0.18 and features["line_density"] < 0.05:
        score += 2
        fired.append("warm room-like surfaces with low top-down structure")
    if (
        features["skin_ratio"] > 0.24
        and features["land_color_mix"] <= 3
        and features["edge_density"] < 0.05
        and (features["neutral_surface_ratio"] > 0.12 or features["smooth_tile_ratio"] > 0.20)
    ):
        score += 2
        fired.append("ground-level person/interior colour composition")
    if features["land_color_mix"] <= 2 and features["colour_diversity"] < 130:
        score += 1
        fired.append("limited top-down land-cover diversity")
    if features["edge_density"] < 0.025 and features["distributed_detail_ratio"] < 0.35:
        score += 1
        fired.append("low distributed aerial texture")

    return score, fired


def _score_document(features: dict[str, float | int]) -> tuple[int, list[str]]:
    score = 0
    fired: list[str] = []

    paper_background = features["light_ratio"] > 0.45 and features["low_saturation_ratio"] > 0.55
    if paper_background:
        score += 2
        fired.append("large uniform paper-like background")

    text_structure = features["text_density_estimate"] > 0.22 and features["text_row_ratio"] > 0.08
    if text_structure:
        score += 2
        fired.append("repeated text-line structure")

    screenshot_text_layout = (
        features["light_ratio"] > 0.45
        and features["low_saturation_ratio"] > 0.60
        and features["text_col_ratio"] > 0.75
        and features["edge_density"] > 0.04
    )
    if screenshot_text_layout:
        score += 2
        fired.append("text-heavy screenshot/document layout")

    card_boundary = features["rectangular_boundary_score"] > 1.35 and features["edge_density"] > 0.012
    if card_boundary:
        score += 1
        fired.append("strong rectangular page/card boundary")

    low_scene_diversity = features["colour_diversity"] < 95 and features["texture_mean"] < 30
    if low_scene_diversity:
        score += 1
        fired.append("low spatial colour/texture diversity")

    document_layout = 0.55 <= features["aspect_ratio"] <= 1.95 and features["line_density"] > 0.06
    if document_layout:
        score += 1
        fired.append("document-like aspect/layout")

    flat_document = features["low_saturation_ratio"] > 0.45 and features["colour_diversity"] < 85
    if flat_document:
        score += 1
        fired.append("flat low-saturation document/card appearance")

    barcode_like_region = features["right_vertical_stripe_ratio"] > 0.10 and features["low_saturation_ratio"] > 0.45
    if barcode_like_region:
        score += 2
        fired.append("barcode-like vertical stripe region")

    photo_box_text_layout = (
        features["left_photo_skin_ratio"] > 0.025
        and features["text_density_estimate"] > 0.25
        and (features["light_ratio"] > 0.25 or features["low_saturation_ratio"] > 0.45)
    )
    if photo_box_text_layout:
        score += 2
        fired.append("photo box plus text/card layout")

    uniform_card_document = (
        features["neutral_surface_ratio"] > 0.55
        and features["text_density_estimate"] > 0.30
        and features["colour_diversity"] < 125
    )
    if uniform_card_document:
        score += 2
        fired.append("uniform card/document background with text")

    dense_id_card_signature = (
        features["text_density_estimate"] > 0.50
        and features["low_saturation_ratio"] > 0.60
        and features["rectangular_boundary_score"] > 1.15
        and features["colour_diversity"] < 115
    )
    if dense_id_card_signature:
        score += 4
        fired.append("dense low-saturation ID/card text layout")

    return score, fired


def _score_aerial(features: dict[str, float | int]) -> tuple[int, list[str]]:
    score = 0
    fired: list[str] = []

    if features["natural_ratio"] > 0.45:
        score += 2
        fired.append("land-cover-like colour regions")
    if features["land_color_mix"] >= 2:
        score += 2
        fired.append("mixed vegetation/urban/water/soil palette")
    if features["entropy"] > 4.0 and features["texture_mean"] > 12:
        score += 1
        fired.append("high texture diversity")
    if features["distributed_detail_ratio"] > 0.35:
        score += 1
        fired.append("detail distributed across frame")
    if features["spatial_variance"] > 12:
        score += 1
        fired.append("land-cover-like spatial variance")
    if features["colour_diversity"] > 120:
        score += 1
        fired.append("high colour diversity")
    if features["white_ratio"] < 0.20:
        score += 1
        fired.append("no dominant white page background")

    return score, fired


def _result(
    status: RelevanceStatus,
    reason: str,
    features: dict[str, float | int],
    doc_score: int,
    aerial_score: int,
    indoor_score: int,
    fired: list[str],
) -> InputValidationResult:
    return InputValidationResult(
        file_integrity="Passed",
        image_relevance=status,
        segmentation="Run" if status == "Suitable" else "Not run",
        reasons=(reason,),
        diagnostics={
            **features,
            "document_score": doc_score,
            "aerial_land_score": aerial_score,
            "indoor_ground_score": indoor_score,
            "thresholds_fired": fired,
        },
    )


def validate_land_image(image: Image.Image) -> InputValidationResult:
    """Classify upload relevance before running the U-Net model.

    The validator is a lightweight multi-stage gate for Streamlit Cloud. It
    combines document/person rejection evidence with positive aerial-land
    evidence. Dense urban geometry alone is not enough to reject an image.
    """

    features = _extract_features(image)
    doc_score, doc_fired = _score_document(features)
    indoor_score, indoor_fired = _score_indoor(features)
    aerial_score, aerial_fired = _score_aerial(features)
    fired = [f"document: {item}" for item in doc_fired] + [f"indoor: {item}" for item in indoor_fired] + [f"aerial: {item}" for item in aerial_fired]

    if features["entropy"] < 1.2 or (features["texture_mean"] < 3 and features["colour_diversity"] < 6):
        return _result("Rejected", "Image is blank or nearly blank.", features, doc_score, aerial_score, indoor_score, fired)

    indoor_veto = (
        indoor_score >= 4
        and features["warm_indoor_ratio"] > 0.22
        and features["line_density"] < 0.08
        and features["land_color_mix"] <= 3
    )
    if indoor_veto or (indoor_score >= 3 and features["warm_indoor_ratio"] > 0.35):
        return _result(
            "Rejected",
            "This appears to be a ground-level indoor image, not satellite or aerial land imagery.",
            features,
            doc_score,
            aerial_score,
            indoor_score,
            fired,
        )

    if features["central_skin_ratio"] > 0.38 or (features["skin_ratio"] > 0.30 and aerial_score < 4):
        return _result("Rejected", "Person or portrait-like content detected.", features, doc_score, aerial_score, indoor_score, fired)

    ground_level_horizon = (
        features["top_sky_ratio"] > 0.70
        and features["edge_density"] < 0.025
        and features["texture_mean"] < 13
        and aerial_score >= 4
    )
    if ground_level_horizon:
        return _result("Uncertain", "The image is ambiguous. Segmentation is disabled by default, but you may run it manually for demonstration.", features, doc_score, aerial_score, indoor_score, fired)

    if indoor_score >= 3:
        if aerial_score <= 6 or features["distributed_detail_ratio"] < 0.60 or features["colour_diversity"] < 140:
            return _result(
                "Rejected",
                "This appears to be a ground-level indoor image, not satellite or aerial land imagery.",
                features,
                doc_score,
                aerial_score,
                indoor_score,
                fired,
            )
        return _result(
            "Uncertain",
            "The image is ambiguous. Segmentation is disabled by default, but you may run it manually for demonstration.",
            features,
            doc_score,
            aerial_score,
            indoor_score,
            fired,
        )

    hard_document_veto = (
        doc_score >= 7
        and (
            features["low_saturation_ratio"] > 0.45
            or features["light_ratio"] > 0.35
            or features["right_vertical_stripe_ratio"] > 0.10
        )
        and features["colour_diversity"] < 140
    )
    if hard_document_veto or (doc_score >= 6 and aerial_score <= 3):
        return _result(
            "Rejected",
            "This appears to be an ID card or document, not satellite or aerial land imagery.",
            features,
            doc_score,
            aerial_score,
            indoor_score,
            fired,
        )

    if doc_score >= 6 and aerial_score <= 4:
        return _result(
            "Rejected",
            "This appears to be an ID card or document, not satellite or aerial land imagery.",
            features,
            doc_score,
            aerial_score,
            indoor_score,
            fired,
        )

    strong_id_card_signature = (
        doc_score >= 8
        and features["text_density_estimate"] > 0.50
        and features["low_saturation_ratio"] > 0.60
        and features["colour_diversity"] < 115
    )
    if strong_id_card_signature:
        return _result(
            "Rejected",
            "This appears to be an ID card or document, not satellite or aerial land imagery.",
            features,
            doc_score,
            aerial_score,
            indoor_score,
            fired,
        )

    screenshot_ui_with_land = features["top_dark_ratio"] > 0.35 and features["light_ratio"] > 0.12 and aerial_score >= 4
    if screenshot_ui_with_land:
        return _result("Uncertain", "The image is ambiguous. Segmentation is disabled by default, but you may run it manually for demonstration.", features, doc_score, aerial_score, indoor_score, fired)

    if indoor_score >= 4 and aerial_score <= 5:
        return _result("Rejected", "This appears to be a ground-level indoor image, not satellite or aerial land imagery.", features, doc_score, aerial_score, indoor_score, fired)

    if indoor_score >= 4 and aerial_score <= 7:
        return _result("Uncertain", "The image is ambiguous. Segmentation is disabled by default, but you may run it manually for demonstration.", features, doc_score, aerial_score, indoor_score, fired)

    if aerial_score >= 6 and doc_score <= 5 and indoor_score <= 2 and aerial_score >= doc_score + 2 and aerial_score >= indoor_score + 4:
        return _result("Suitable", "Multiple aerial/land characteristics detected.", features, doc_score, aerial_score, indoor_score, fired)

    if aerial_score >= 5 and aerial_score > doc_score + 1 and indoor_score <= 1:
        return _result("Suitable", "Aerial/land evidence is stronger than document-like evidence.", features, doc_score, aerial_score, indoor_score, fired)

    if doc_score >= 5 and aerial_score <= 4:
        return _result("Rejected", "This appears to be an ID card or document, not satellite or aerial land imagery.", features, doc_score, aerial_score, indoor_score, fired)

    if aerial_score < 3 and doc_score < 4:
        return _result("Rejected", "Image contains insufficient aerial/land characteristics.", features, doc_score, aerial_score, indoor_score, fired)

    return _result("Uncertain", "The image is ambiguous. Segmentation is disabled by default, but you may run it manually for demonstration.", features, doc_score, aerial_score, indoor_score, fired)
