from __future__ import annotations

import numpy as np

from .config import CLASS_NAMES, NUM_CLASSES


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    true = y_true.reshape(-1).astype(np.int64)
    pred = y_pred.reshape(-1).astype(np.int64)
    valid = (true >= 0) & (true < num_classes)
    encoded = num_classes * true[valid] + pred[valid]
    return np.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)


def segmentation_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES) -> dict:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    total = cm.sum()

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    iou = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp), where=(tp + fp + fn) != 0)
    dice = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) != 0)

    present = cm.sum(axis=1) > 0
    return {
        "pixel_accuracy": float(tp.sum() / total) if total else 0.0,
        "mean_iou": float(iou[present].mean()) if present.any() else 0.0,
        "mean_dice": float(dice[present].mean()) if present.any() else 0.0,
        "macro_precision": float(precision[present].mean()) if present.any() else 0.0,
        "macro_recall": float(recall[present].mean()) if present.any() else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "per_class": {
            CLASS_NAMES[idx]: {
                "iou": float(iou[idx]),
                "dice": float(dice[idx]),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support_pixels": int(cm[idx].sum()),
            }
            for idx in range(num_classes)
        },
        "confusion_matrix": cm.astype(int).tolist(),
    }

