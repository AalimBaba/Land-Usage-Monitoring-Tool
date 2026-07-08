from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from land_usage.config import CLASS_NAMES, PALETTE, REPO_URL
from land_usage.inference import ModelUnavailableError, load_model, predict_image


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "unet_model.h5"
METRICS_PATH = ROOT / "metrics.json"


st.set_page_config(
    page_title="Land Usage Monitoring Tool",
    page_icon="L",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ink: #10211f;
          --muted: #516461;
          --line: #d9e1dc;
          --field: #f4f8f6;
          --accent: #1f7a5c;
          --accent-2: #295a8d;
          --warn: #a45f19;
        }
        .stApp {
          color: var(--ink);
          background:
            linear-gradient(180deg, rgba(238,246,242,.92), rgba(255,255,255,.96) 42%),
            url("https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?auto=format&fit=crop&w=1800&q=80");
          background-size: cover;
          background-attachment: fixed;
        }
        [data-testid="stHeader"] { background: rgba(255,255,255,.72); backdrop-filter: blur(14px); }
        .block-container { padding-top: 2.8rem; max-width: 1180px; }
        .hero {
          min-height: 430px;
          display: flex;
          flex-direction: column;
          justify-content: center;
          padding: 3.5rem 0 2rem;
        }
        .eyebrow {
          color: var(--accent);
          font-weight: 800;
          letter-spacing: .08em;
          text-transform: uppercase;
          font-size: .78rem;
        }
        h1 {
          font-size: clamp(2.4rem, 8vw, 5.9rem);
          line-height: .94;
          letter-spacing: 0;
          margin: .4rem 0 1rem;
          max-width: 900px;
        }
        .lead {
          max-width: 720px;
          color: #263936;
          font-size: clamp(1rem, 2vw, 1.25rem);
          line-height: 1.7;
        }
        .pill-row { display: flex; flex-wrap: wrap; gap: .65rem; margin-top: 1.3rem; }
        .pill {
          border: 1px solid rgba(16,33,31,.16);
          background: rgba(255,255,255,.74);
          border-radius: 999px;
          padding: .55rem .78rem;
          color: #203532;
          font-weight: 700;
          font-size: .86rem;
        }
        .metric-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: .8rem;
        }
        .metric {
          background: var(--field);
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: .9rem;
        }
        .metric b { display: block; font-size: .8rem; color: var(--muted); margin-bottom: .35rem; }
        .metric span { font-size: 1.35rem; font-weight: 800; color: var(--ink); }
        .legend {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: .45rem .8rem;
        }
        .legend-item { display: flex; align-items: center; gap: .5rem; font-size: .9rem; color: #344844; }
        .swatch { width: 14px; height: 14px; border-radius: 3px; border: 1px solid rgba(0,0,0,.12); }
        .note {
          border-left: 4px solid var(--warn);
          background: #fff8ef;
          padding: .9rem 1rem;
          border-radius: 6px;
          color: #6f4316;
        }
        .footer-link a { color: var(--accent-2); font-weight: 800; text-decoration: none; }
        [data-testid="stVerticalBlockBorderWrapper"] {
          background: rgba(255,255,255,.9);
          box-shadow: 0 18px 50px rgba(28,52,46,.08);
        }
        @media (max-width: 700px) {
          .block-container { padding-left: 1rem; padding-right: 1rem; }
          .hero { min-height: 380px; padding-top: 2.5rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_value(metrics: dict, key: str) -> str:
    value = metrics.get(key)
    if value is None:
        return "Not evaluated"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_distribution(distribution: dict[str, float], confidence: float) -> None:
    st.subheader("Predicted Class Distribution")
    for name, pct in distribution.items():
        st.progress(min(max(pct / 100, 0), 1), text=f"{name}: {pct:.1f}%")
    st.caption(f"Mean pixel confidence: {confidence:.1f}%. Confidence is model softmax confidence, not real-world accuracy.")


def render_legend() -> None:
    rows = []
    for idx, name in enumerate(CLASS_NAMES):
        color = "#{:02x}{:02x}{:02x}".format(*PALETTE[idx])
        rows.append(f'<div class="legend-item"><span class="swatch" style="background:{color}"></span>{idx}: {name}</div>')
    st.markdown('<div class="legend">' + "".join(rows) + "</div>", unsafe_allow_html=True)


css()
metrics = load_metrics()

st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">Semantic Segmentation - Satellite/Land Imagery</div>
      <h1>Land Usage Monitoring Tool</h1>
      <p class="lead">
        Upload a land or satellite image and run a U-Net segmentation model that estimates pixel-level land-use regions.
        The project is designed for honest monitoring demos: it shows predictions, class distribution, confidence, and known limitations without claiming perfect accuracy.
      </p>
      <div class="pill-row">
        <span class="pill">7-class segmentation</span>
        <span class="pill">Real model inference</span>
        <span class="pill">Deployment-ready Streamlit app</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.header("Try the Model")
    st.write(
        "Upload a PNG or JPEG land/satellite image. The app resizes it to the model's 64 x 64 input, "
        "runs the bundled U-Net, and displays the segmentation overlay and class mix."
    )
    upload = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)

    if upload is None:
        st.info("Choose an image to run inference.")
    else:
        try:
            image = Image.open(upload).convert("RGB")
            model = load_model(MODEL_PATH)
            with st.spinner("Running segmentation model..."):
                result = predict_image(model, image)

            left, mid, right = st.columns(3)
            left.image(image, caption="Uploaded image", use_column_width=True)
            mid.image(result.mask_image, caption="Predicted land-use map", use_column_width=True)
            right.image(result.overlay_image, caption="Prediction overlay", use_column_width=True)
            render_distribution(result.class_distribution, result.mean_confidence)
            st.markdown("**Legend**")
            render_legend()
        except UnidentifiedImageError:
            st.error("That file is not a valid image. Please upload a PNG or JPEG image.")
        except ModelUnavailableError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Prediction failed gracefully: {exc}")

col_a, col_b = st.columns(2)
with col_a:
    with st.container(border=True):
        st.header("How It Works")
        st.write(
            "This is a semantic segmentation task. Each pixel is assigned one land-use class. "
            "Training pairs RGB `.png` images with single-channel `.tif` masks using matching file names."
        )
        st.write(
            "The improved pipeline keeps image/mask resizing aligned, remaps mask labels consistently, "
            "uses train/validation/test splits, and evaluates segmentation quality beyond pixel accuracy."
        )

with col_b:
    with st.container(border=True):
        st.header("Model Improvements")
        st.write(
            "The repository now includes augmentation, Dice + focal cross-entropy loss, early stopping, "
            "best-model checkpointing, learning-rate scheduling, and reusable metrics. "
            "A transfer-learning U-Net path can be added later, but the default remains simple to run."
        )

with st.container(border=True):
    st.header("Evaluation Metrics")
    st.markdown(
        f"""
        <div class="metric-grid">
          <div class="metric"><b>Pixel Accuracy</b><span>{metric_value(metrics, "pixel_accuracy")}</span></div>
          <div class="metric"><b>Mean IoU</b><span>{metric_value(metrics, "mean_iou")}</span></div>
          <div class="metric"><b>Mean Dice</b><span>{metric_value(metrics, "mean_dice")}</span></div>
          <div class="metric"><b>Macro F1</b><span>{metric_value(metrics, "macro_f1")}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not metrics:
        st.markdown(
            '<p class="note">No evaluation dataset is included in this repository, so this live build does not display fabricated metrics. Run <code>python evaluate_model.py --data-dir "path/to/SEN-2 LULC"</code> after downloading the dataset to generate <code>metrics.json</code>.</p>',
            unsafe_allow_html=True,
        )

cols = st.columns(3)
with cols[0]:
    with st.container(border=True):
        st.header("Tech Stack")
        st.write("Python, Streamlit, TensorFlow/Keras, NumPy, Pillow, scikit-learn, Matplotlib, Rasterio.")
with cols[1]:
    with st.container(border=True):
        st.header("Limitations")
        st.write(
            "The bundled model is small and trained at 64 x 64 resolution. It may miss fine boundaries, "
            "confuse visually similar land types, and should not be used for policy or safety decisions without stronger validation."
        )
with cols[2]:
    with st.container(border=True):
        st.header("GitHub Repository")
        st.markdown(f'<div class="footer-link"><a href="{REPO_URL}">View source code</a></div>', unsafe_allow_html=True)
        st.write("Live demo link belongs in the README after deployment.")
