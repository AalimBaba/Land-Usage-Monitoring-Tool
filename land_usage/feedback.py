from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


FEEDBACK_DIR = Path("feedback")
FEEDBACK_PATH = FEEDBACK_DIR / "feedback_log.jsonl"


def image_hash(image: Image.Image) -> str:
    resized = image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
    return hashlib.sha256(resized.tobytes()).hexdigest()


def feedback_record(
    *,
    uploaded_filename: str | None,
    image: Image.Image,
    validation_status: str,
    segmentation_status: str,
    dominant_class: str | None,
    pixel_distribution: dict[str, float] | None,
    mean_softmax_confidence: float | None,
    feedback_category: str,
    user_note: str,
    expected_input_type: str,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uploaded_filename": uploaded_filename,
        "validation_status": validation_status,
        "segmentation_status": segmentation_status,
        "predicted_dominant_class": dominant_class,
        "predicted_pixel_distribution": pixel_distribution,
        "mean_softmax_confidence": mean_softmax_confidence,
        "user_feedback_category": feedback_category,
        "user_note": user_note,
        "expected_input_type": expected_input_type,
        "image_hash": image_hash(image),
    }


def append_feedback(record: dict[str, Any], path: Path = FEEDBACK_PATH) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return True
    except OSError:
        return False


def records_to_jsonl(records: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(record, sort_keys=True) for record in records)
