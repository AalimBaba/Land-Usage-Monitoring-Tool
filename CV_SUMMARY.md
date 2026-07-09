# CV Summary

## Resume Bullets

- Built and deployed a Streamlit-based land usage monitoring web app using U-Net inference to generate segmentation masks, overlays, class distributions, and confidence estimates from uploaded satellite imagery.
- Implemented validation benchmark and evaluation pipeline; segmentation accuracy pending labelled held-out dataset.

## Project Description

Land Usage Monitoring Tool is a web-based semantic segmentation project for estimating land-use regions from satellite or land imagery using a compact TensorFlow/Keras U-Net model.

## Tech Stack

Python, Streamlit, TensorFlow/Keras, U-Net, NumPy, Pillow, Scikit-learn, Matplotlib, Pytest, Streamlit Cloud.

## Metrics

Exact segmentation metrics achieved from real evaluation: **not available yet because the labelled held-out satellite image/mask dataset is not included in this repository.** Run `python evaluate_model.py --data-dir "path/to/SEN-2 LULC" --model unet_model.h5` to generate `metrics.json`, then add the real pixel accuracy, mIoU, Dice, precision, recall, F1-score, and per-class IoU values here.

Input-validation benchmark: `13/13` built-in benchmark cases passed with `0` false accepts and `0` false rejects. This measures upload validation only, not U-Net segmentation accuracy.

## Live Demo

https://land-usage-monitoring-tool.streamlit.app

## GitHub

https://github.com/AalimBaba/Land-Usage-Monitoring-Tool
