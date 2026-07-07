# Land Usage Monitoring Tool

A portfolio-ready web app for monitoring land usage from satellite or land imagery with semantic segmentation. The app lets a user upload an image, runs a bundled U-Net model, and returns a predicted land-use map, overlay, class distribution, and model confidence.

Live demo: **TBD after deployment**

GitHub: https://github.com/AalimBaba/Land-Usage-Monitoring-Tool

## Features

- Upload PNG/JPEG satellite or land images.
- Run real TensorFlow/Keras U-Net inference from `unet_model.h5`.
- Display predicted segmentation mask, overlay, class distribution, and mean softmax confidence.
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

## Why the Original Accuracy Was Limited

The original code used Windows-only absolute paths, trained on only the first 50 image/mask pairs, evaluated only the first 20 test pairs, relied mainly on pixel accuracy, used no augmentation, had no best-model checkpointing, and used mismatched inference labels in the web app. Pixel accuracy can look acceptable even when minority classes and boundaries are poor, so the improved pipeline reports segmentation metrics such as mIoU and Dice.

## Training

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
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
streamlit run app.py
```

Upload a PNG/JPEG image in the "Try the Model" area. The app resizes the image to `64 x 64`, runs the U-Net, and shows the predicted segmentation. Confidence is the model's mean softmax confidence, not a guarantee of real-world correctness.

## Deployment

Recommended deployment: **Streamlit Community Cloud**.

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud and create a new app.
3. Select `AalimBaba/Land-Usage-Monitoring-Tool`.
4. Set the main file path to `app.py`.
5. Deploy.
6. Replace the live demo placeholder in this README and `CV_SUMMARY.md` with the deployed URL.

Alternative Docker deployment:

```bash
docker build -t land-usage-monitoring .
docker run -p 8501:8501 land-usage-monitoring
```

The app does not require API keys or environment variables for local inference. The model path is relative to the repository root.

## Tech Stack

Python, Streamlit, TensorFlow/Keras, NumPy, Pillow, Matplotlib, scikit-learn, Rasterio, Docker.

## Limitations and Ethical Note

The current model is small and operates at low resolution. It may confuse similar land classes, miss small structures, and produce coarse boundaries. It should be treated as an educational and portfolio ML project, not as a source of legal, environmental, agricultural, or policy decisions. Any public use should include dataset provenance, geographic coverage, validation results, and human review.

## Screenshots

Add screenshots after deployment:

- Home page / hero
- Upload area
- Prediction result
- Mobile layout

## Author

Aalim Baba
