# CV Summary

## Resume Bullets

- Built and deployed a Streamlit land-usage monitoring app that performs real U-Net semantic segmentation inference on uploaded satellite/land images, returning predicted masks, overlays, class distributions, and confidence summaries.
- Refactored the ML workflow into a reproducible training/evaluation pipeline with aligned preprocessing, data augmentation, train/validation/test splitting, Dice/focal loss, checkpointing, early stopping, learning-rate scheduling, and segmentation metrics.

## Project Description

Land Usage Monitoring Tool is a web-based semantic segmentation project for estimating land-use regions from satellite or land imagery using a compact TensorFlow/Keras U-Net model.

## Tech Stack

Python, Streamlit, TensorFlow/Keras CPU runtime, NumPy, Pillow, Matplotlib, scikit-learn, Rasterio, Docker.

## Metrics

Exact metrics achieved from real evaluation: **not available yet because the evaluation dataset is not included in this repository.** Run `python evaluate_model.py --data-dir "path/to/SEN-2 LULC" --model unet_model.h5` to generate `metrics.json`, then add the real pixel accuracy, mIoU, Dice, precision, recall, F1-score, and per-class IoU values here.

## Live Demo

TBD after deployment.

## GitHub

https://github.com/AalimBaba/Land-Usage-Monitoring-Tool
