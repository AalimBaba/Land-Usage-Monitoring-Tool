from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from land_usage.config import MODEL_INPUT_SIZE, NUM_CLASSES
from land_usage.data import discover_dataset, load_pairs
from land_usage.metrics import segmentation_metrics
from land_usage.modeling import build_unet, combined_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a U-Net land-use segmentation model.")
    parser.add_argument("--data-dir", required=True, type=Path, help="Dataset folder containing split folders or images/masks.")
    parser.add_argument("--output", default=Path("unet_model.h5"), type=Path, help="Best model output path.")
    parser.add_argument("--metrics-output", default=Path("metrics.json"), type=Path)
    parser.add_argument("--epochs", default=60, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def make_dataset(images: np.ndarray, masks: np.ndarray, batch_size: int, training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((images, masks))
    if training:
        ds = ds.shuffle(len(images), seed=42, reshuffle_each_iteration=True)
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def augment(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask[..., None])[..., 0]
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask[..., None])[..., 0]
    image = tf.image.random_brightness(image, max_delta=0.08)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return tf.clip_by_value(image, 0.0, 1.0), mask


def plot_history(history: tf.keras.callbacks.History, output_path: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history.get("sparse_categorical_accuracy", []), label="train")
    plt.plot(history.history.get("val_sparse_categorical_accuracy", []), label="validation")
    plt.title("Pixel accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    train_pairs, val_pairs, test_pairs = discover_dataset(args.data_dir, seed=args.seed)
    print(f"Pairs: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")

    train_images, train_masks = load_pairs(train_pairs, MODEL_INPUT_SIZE)
    val_images, val_masks = load_pairs(val_pairs, MODEL_INPUT_SIZE)
    test_images, test_masks = load_pairs(test_pairs, MODEL_INPUT_SIZE)

    model = build_unet()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=combined_loss,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="sparse_categorical_accuracy")],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.output, monitor="val_loss", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger("training_log.csv"),
    ]

    history = model.fit(
        make_dataset(train_images, train_masks, args.batch_size, training=True),
        validation_data=make_dataset(val_images, val_masks, args.batch_size, training=False),
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    plot_history(history, Path("training_plot.png"))
    best_model = tf.keras.models.load_model(args.output, custom_objects={"combined_loss": combined_loss}, compile=False)
    predictions = np.argmax(best_model.predict(test_images, batch_size=args.batch_size, verbose=1), axis=-1)
    metrics = segmentation_metrics(test_masks, predictions, NUM_CLASSES)
    metrics["dataset"] = {"train_pairs": len(train_pairs), "validation_pairs": len(val_pairs), "test_pairs": len(test_pairs)}
    args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({key: metrics[key] for key in ["pixel_accuracy", "mean_iou", "mean_dice", "macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
