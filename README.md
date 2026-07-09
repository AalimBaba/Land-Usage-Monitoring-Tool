# Land Usage Monitoring Tool

A portfolio-ready web app for monitoring land usage from satellite or land imagery with semantic segmentation. The app lets a user upload an image, runs a bundled U-Net model, and returns a predicted land-use map, overlay, class distribution, and model confidence.

Live demo: **https://land-usage-monitoring-tool.streamlit.app**

GitHub: https://github.com/AalimBaba/Land-Usage-Monitoring-Tool

## Features

- Upload PNG/JPEG satellite or land images.
- Reject obvious out-of-distribution uploads such as documents, certificates, screenshots, portraits, blank images, and simple graphics before segmentation.
- Run real TensorFlow/Keras U-Net inference from `unet_model.h5`.
- Display predicted segmentation mask, overlay, class distribution, and mean softmax confidence.
- Collect metadata-only user feedback for rejected, uncertain, and completed predictions, with a session download option and no raw image storage by default.
- Gracefully handle invalid images and missing model files.
- Keep training, evaluation, and inference preprocessing consistent.
- Report honest metrics only from real evaluation runs.

## Architecture

```text
Streamlit app
  app.py
    -> land_usage.inference.load_model()
    -> land_usage.inference.predict_image()
    -> predicted mask / overlay / class percentages

Training and evaluation
  train_model.py
    -> land_usage.data
    -> land_usage.modeling
    -> best checkpoint + training logs + metrics.json
  evaluate_model.py
    -> test split predictions
    -> mIoU, Dice, pixel accuracy, precision, recall, F1, confusion matrix
```

## Dataset and Model

This project is a **semantic segmentation** task, not simple image classification. Each pixel in an input image is assigned one of seven land-use classes.

Expected dataset formats:

```text
SEN-2 LULC/
  train_images/*.png
  train_masks/*.tif
  val_images/*.png
  val_masks/*.tif
  test_images/*.png
  test_masks/*.tif
```

or:

```text
dataset/
  images/*
  masks/*
```

For unsplit datasets, the scripts create reproducible train/validation/test splits with a fixed seed.

The bundled model is a compact U-Net with input shape `64 x 64 x 3` and seven output classes. It is suitable for demonstration and experimentation, not production-grade monitoring.

The deployed app includes a lightweight multi-stage pre-inference validation layer. This gate checks file integrity, scores document/card evidence, scores indoor/ground-photo evidence, scores aerial/land evidence, and returns `Suitable`, `Uncertain`, or `Rejected` before U-Net inference. It uses image statistics, text-density cues, rectangular card/document cues, photo-box and barcode-like layout cues, skin-tone layout, blank-image detection, smooth indoor-surface cues, texture, colour diversity, spatial variance, and land-colour patterns. It is intentionally lightweight for Streamlit Cloud and should be treated as an explainable heuristic, not a calibrated semantic classifier.

## Why the Original Accuracy Was Limited

The original code used Windows-only absolute paths, trained on only the first 50 image/mask pairs, evaluated only the first 20 test pairs, relied mainly on pixel accuracy, used no augmentation, had no best-model checkpointing, and used mismatched inference labels in the web app. Pixel accuracy can look acceptable even when minority classes and boundaries are poor, so the improved pipeline reports segmentation metrics such as mIoU and Dice.

## Training

Create an environment and install dependencies:

```bash
pip install -r requirements-dev.txt
```

Train the model:

```bash
python train_model.py --data-dir "path/to/SEN-2 LULC" --epochs 60 --batch-size 8
```

Training includes:

- aligned image/mask resizing;
- random flips, brightness, and contrast augmentation;
- Dice + sparse categorical cross-entropy + focal loss;
- early stopping;
- learning-rate scheduling;
- best-model checkpointing;
- CSV training logs.

The best model is saved to `unet_model.h5` by default.

## Evaluation

Run evaluation on the test split:

```bash
python evaluate_model.py --data-dir "path/to/SEN-2 LULC" --model unet_model.h5
```

This writes `metrics.json` with:

- pixel accuracy;
- mean IoU;
- mean Dice;
- per-class IoU;
- precision;
- recall;
- F1-score;
- confusion matrix.

Current metrics: **not included in this repository because the dataset is not committed.** Do not add CV metrics until `metrics.json` is generated from a real test split.

## Inference

Run the web app locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload a PNG/JPEG image in the "Try the Model" area. The app resizes the image to `64 x 64`, runs the U-Net, and shows the predicted segmentation. Confidence is the model's mean softmax confidence, not a guarantee of real-world correctness.

The feedback panel lets users report false rejections, false acceptances, or incorrect-looking land predictions. On Streamlit Community Cloud, app-local files are temporary and should not be treated as durable storage. The app therefore keeps feedback in the current Streamlit session and exposes a **Download feedback log** button after feedback is submitted. It also attempts a best-effort metadata-only JSONL write to `feedback/feedback_log.jsonl` when the runtime filesystem allows it, but users should download the JSONL log if they need to keep it. Uploaded images are not stored by default.

## Deployment

Recommended deployment: **Streamlit Community Cloud**.

Use these exact settings:

| Setting | Value |
| --- | --- |
| Repository | `AalimBaba/Land-Usage-Monitoring-Tool` |
| Branch | `main` |
| Main file path | `app.py` |
| Python version | `3.11` |
| Platform | Streamlit Community Cloud |

Deployment steps:

1. Open Streamlit Community Cloud and create a new app.
2. Select `AalimBaba/Land-Usage-Monitoring-Tool`.
3. Select branch `main`.
4. Set the main file path to `app.py`.
5. Open **Advanced settings** and choose Python `3.11`.
6. Deploy.
7. Verify the live app opens, accepts uploads, and runs inference before adding the link to a CV.

Important: Streamlit Community Cloud selects Python version in the app's Advanced settings. If an existing app was created with the wrong Python version, delete and redeploy it with Python `3.11`; rebooting alone may keep the old Python runtime.

Alternative Docker deployment:

```bash
docker build -t land-usage-monitoring .
docker run -p 8501:8501 land-usage-monitoring
```

The app does not require API keys or environment variables for local inference. The model path is relative to the repository root.

## Tech Stack

Python, Streamlit, TensorFlow/Keras, U-Net, NumPy, Pillow, Scikit-learn, Matplotlib, Pytest, Streamlit Cloud.

## Limitations and Ethical Note

The bundled demo model is small and operates at low `64 x 64` resolution. Predictions depend on dataset quality and may confuse similar land classes, miss small structures, or produce coarse boundaries. The input relevance gate rejects many obvious non-land images, but heuristic validation can still false-reject valid imagery or miss unusual irrelevant images. The app is not suitable for official land surveys, legal decisions, environmental policy, agricultural planning, or safety-critical monitoring. Verified accuracy requires a labelled held-out evaluation dataset. Larger datasets, higher-resolution training, and stronger architectures such as DeepLabV3+, U-Net++, or SegFormer can improve performance.

## Screenshots

Add screenshots after deployment:

- Home page / hero
- Upload area
- Prediction result
- Mobile layout

## Author

Aalim Baba
