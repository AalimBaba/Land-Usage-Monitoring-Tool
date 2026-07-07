from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError

from land_usage.config import MODEL_INPUT_SIZE, NUM_CLASSES
from land_usage.inference import predict_image
from land_usage.metrics import segmentation_metrics


class FakeModel:
    def predict(self, batch, verbose=0):
        height, width = MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0]
        probabilities = np.zeros((1, height, width, NUM_CLASSES), dtype=np.float32)
        probabilities[..., 2] = 0.8
        probabilities[..., 1] = 0.2
        return probabilities


def test_predict_image_returns_mask_distribution_and_confidence():
    image = Image.fromarray(np.full((128, 128, 3), 120, dtype=np.uint8))

    result = predict_image(FakeModel(), image)

    assert result.mask.shape == (MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0])
    assert round(sum(result.class_distribution.values()), 4) == 100.0
    assert result.class_distribution["Vegetation"] == 100.0
    assert round(result.mean_confidence, 1) == 80.0


def test_invalid_upload_bytes_raise_image_error():
    invalid_file = BytesIO(b"not an image")

    try:
        Image.open(invalid_file).convert("RGB")
    except UnidentifiedImageError:
        return

    raise AssertionError("invalid image bytes should raise UnidentifiedImageError")


def test_segmentation_metrics_are_computed_from_confusion_matrix():
    y_true = np.array([[0, 1], [1, 2]])
    y_pred = np.array([[0, 1], [2, 2]])

    metrics = segmentation_metrics(y_true, y_pred, num_classes=NUM_CLASSES)

    assert metrics["pixel_accuracy"] == 0.75
    assert "Vegetation" in metrics["per_class"]
    assert len(metrics["confusion_matrix"]) == NUM_CLASSES
