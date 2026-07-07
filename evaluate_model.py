from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from land_usage.config import CLASS_NAMES, MODEL_INPUT_SIZE, NUM_CLASSES
from land_usage.data import discover_dataset, load_pairs
from land_usage.inference import colorize_mask
from land_usage.metrics import segmentation_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a land-use segmentation model on the test split.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--model", default=Path("unet_model.h5"), type=Path)
    parser.add_argument("--output", default=Path("metrics.json"), type=Path)
    parser.add_argument("--plot-output", default=Path("test_predictions.png"), type=Path)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def save_prediction_plot(images: np.ndarray, masks: np.ndarray, preds: np.ndarray, output: Path) -> None:
    count = min(5, len(images))
    if count == 0:
        return
    plt.figure(figsize=(12, 4 * count))
    for idx in range(count):
        plt.subplot(count, 3, idx * 3 + 1)
        plt.title("Image")
        plt.imshow(images[idx])
        plt.axis("off")
        plt.subplot(count, 3, idx * 3 + 2)
        plt.title("True mask")
        plt.imshow(colorize_mask(masks[idx]))
        plt.axis("off")
        plt.subplot(count, 3, idx * 3 + 3)
        plt.title("Predicted mask")
        plt.imshow(colorize_mask(preds[idx]))
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=160)


def main() -> None:
    args = parse_args()
    _, _, test_pairs = discover_dataset(args.data_dir, seed=args.seed)
    test_images, test_masks = load_pairs(test_pairs, MODEL_INPUT_SIZE)

    model = tf.keras.models.load_model(args.model, compile=False)
    probabilities = model.predict(test_images, batch_size=args.batch_size, verbose=1)
    predictions = np.argmax(probabilities, axis=-1)
    metrics = segmentation_metrics(test_masks, predictions, NUM_CLASSES)
    metrics["dataset"] = {"test_pairs": len(test_pairs)}
    metrics["class_names"] = CLASS_NAMES

    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_prediction_plot(test_images, test_masks, predictions, args.plot_output)

    print("Evaluation complete")
    for key in ("pixel_accuracy", "mean_iou", "mean_dice", "macro_precision", "macro_recall", "macro_f1"):
        print(f"{key}: {metrics[key]:.4f}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
