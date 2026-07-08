from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from land_usage.config import CLASS_NAMES, PALETTE, REPO_URL
from land_usage.inference import ModelUnavailableError, load_model, predict_image
from land_usage.validation import InputValidationResult, validate_land_image


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
          --panel: rgba(255,255,255,.93);
          --panel-strong: rgba(255,255,255,.98);
          --page-soft: rgba(238,246,242,.92);
          --page-base: rgba(255,255,255,.96);
          --accent: #1f7a5c;
          --accent-2: #295a8d;
          --warn: #a45f19;
          --shadow: 0 18px 50px rgba(28,52,46,.10);
          --upload: #edf6f1;
        }
        .stApp {
          color: var(--ink);
          background:
            linear-gradient(180deg, var(--page-soft), var(--page-base) 42%),
            url("https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?auto=format&fit=crop&w=1800&q=80");
          background-size: cover;
          background-attachment: fixed;
        }
        [data-testid="stHeader"] { background: color-mix(in srgb, var(--panel) 76%, transparent); backdrop-filter: blur(14px); }
        .block-container {
          padding-top: 2.4rem;
          padding-left: clamp(1rem, 3vw, 3.5rem);
          padding-right: clamp(1rem, 3vw, 3.5rem);
          max-width: 1540px;
        }
        .hero {
          min-height: 390px;
          display: flex;
          flex-direction: column;
          justify-content: center;
          padding: 3rem 0 2rem;
        }
        h1 {
          font-size: clamp(2.5rem, 7vw, 6rem);
          line-height: .95;
          letter-spacing: 0;
          margin: 0 0 1rem;
          max-width: 980px;
        }
        .lead {
          max-width: 880px;
          color: #263936;
          font-size: clamp(1rem, 2vw, 1.25rem);
          line-height: 1.7;
        }
        .pill-row { display: flex; flex-wrap: wrap; gap: .65rem; margin-top: 1.3rem; }
        .pill {
          border: 1px solid rgba(16,33,31,.16);
          background: color-mix(in srgb, var(--panel-strong) 82%, transparent);
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
          background: var(--panel);
          border-color: var(--line);
          box-shadow: var(--shadow);
        }
        [data-testid="stFileUploaderDropzone"] {
          background: var(--upload);
          border: 1px dashed color-mix(in srgb, var(--accent) 60%, var(--line));
          border-radius: 10px;
        }
        [data-testid="stImage"] img {
          border-radius: 8px;
          border: 1px solid var(--line);
          background: var(--field);
        }
        div[data-testid="stMetric"] {
          background: var(--field);
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: .7rem .8rem;
        }
        @media (prefers-color-scheme: dark) {
          :root {
            --ink: #edf7f3;
            --muted: #b8c9c4;
            --line: #314842;
            --field: #172421;
            --panel: rgba(16,29,26,.94);
            --panel-strong: rgba(23,36,33,.98);
            --page-soft: rgba(8,17,16,.88);
            --page-base: rgba(8,13,12,.97);
            --accent: #6ed3a6;
            --accent-2: #9cc9ff;
            --warn: #efb15f;
            --shadow: 0 18px 54px rgba(0,0,0,.38);
            --upload: #13231f;
          }
          .lead, .legend-item, .pill { color: var(--ink); }
          .note { background: #2a2115; color: #ffdca8; }
          h1, h2, h3, p, label, span { color: var(--ink); }
        }
        @media (max-width: 700px) {
          .block-container { padding-left: 1rem; padding-right: 1rem; }
          .hero { min-height: 340px; padding-top: 2rem; }
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
    st.subheader("Predicted Pixel Distribution")
    st.write(
        "These percentages represent the share of image pixels assigned to each predicted land class. "
        "They are not model accuracy or confidence scores."
    )
    for name, pct in distribution.items():
        st.progress(min(max(pct / 100, 0), 1), text=f"{name}: {pct:.1f}%")
    st.caption(f"Mean pixel confidence: {confidence:.1f}%. Confidence is model softmax confidence, not real-world accuracy.")


def render_validation_status(validation: InputValidationResult, segmentation: str | None = None) -> None:
    st.markdown("**INPUT VALIDATION**")
    cols = st.columns(3)
    cols[0].metric("File integrity", validation.file_integrity)
    cols[1].metric("Image relevance", validation.image_relevance)
    cols[2].metric("Segmentation", segmentation or validation.segmentation)
    if validation.reasons:
        st.caption(" ".join(validation.reasons))


def dominant_class(distribution: dict[str, float]) -> tuple[str, float]:
    if not distribution:
        return "Unknown", 0.0
    return max(distribution.items(), key=lambda item: item[1])


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
      <h1>Land Usage Monitoring Tool</h1>
      <p class="lead">
        A deployed Streamlit web app for monitoring land usage from satellite or land imagery. Upload an image,
        run real TensorFlow/Keras U-Net inference, and review a predicted land-use mask, segmentation overlay,
        class distribution, and confidence estimate in one portfolio-ready workflow.
      </p>
      <div class="pill-row">
        <span class="pill">7-class semantic segmentation</span>
        <span class="pill">U-Net model inference</span>
        <span class="pill">Overlay + class distribution</span>
        <span class="pill">Live on Streamlit Cloud</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.header("Try the Model")
    st.write(
        "Upload a clear PNG or JPEG satellite/land image, such as a crop, neighborhood, water body, field, "
        "or mixed land-cover scene. The bundled demo model resizes every image to its expected 64 x 64 input, "
        "runs U-Net inference, and displays the segmentation overlay and class mix."
    )
    st.caption(
        "This is a compact demo model, not a survey-grade system. Results are best interpreted as a portfolio "
        "example of an end-to-end segmentation workflow. Documents, certificates, portraits, screenshots, blank images, "
        "and simple graphics are rejected before U-Net inference."
    )
    upload = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)

    if upload is None:
        st.info("Choose an image to run inference.")
    else:
        try:
            image = Image.open(upload).convert("RGB")
            validation = validate_land_image(image)

            should_run_segmentation = validation.is_suitable
            if validation.is_rejected:
                render_validation_status(validation)
                st.error(
                    "This image does not appear to be satellite or aerial land imagery. "
                    "Please upload a suitable land or satellite image."
                )
                should_run_segmentation = False
            elif validation.is_uncertain:
                render_validation_status(validation)
                st.warning(
                    "This image may not be satellite or aerial land imagery. Run segmentation only if the upload is actually a land, aerial, or remote-sensing image."
                )
                should_run_segmentation = st.checkbox("Run segmentation anyway for this uncertain image")

            if not should_run_segmentation:
                st.stop()

            model = load_model(MODEL_PATH)
            with st.spinner("Running segmentation model..."):
                result = predict_image(model, image)
            render_validation_status(validation, segmentation="Run")

            left, mid, right = st.columns(3)
            left.image(image, caption="Uploaded Image", use_column_width=True)
            mid.image(result.mask_image, caption="Predicted Land-Use Mask", use_column_width=True)
            right.image(result.overlay_image, caption="Segmentation Overlay", use_column_width=True)
            dominant_name, dominant_pct = dominant_class(result.class_distribution)
            st.success(
                f"The model estimates the dominant land-use class as {dominant_name} "
                f"with {dominant_pct:.1f}% predicted pixel share."
            )
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
            '<div class="note"><strong>Held-out evaluation dataset not included in repository.</strong><br>'
            'The metrics pipeline is implemented, but this live demo does not fabricate scores. '
            'Run <code>python evaluate_model.py --data-dir "path/to/SEN-2 LULC" --model unet_model.h5</code> '
            'with labelled image/mask pairs to generate <code>metrics.json</code>. Supported metrics include '
            'Pixel Accuracy, Mean IoU, Mean Dice, Macro F1, per-class IoU, precision, recall, F1-score, and a confusion matrix.</div>',
            unsafe_allow_html=True,
        )

cols = st.columns(3)
with cols[0]:
    with st.container(border=True):
        st.header("Tech Stack")
        st.write("Python, Streamlit, TensorFlow/Keras, U-Net, NumPy, Pillow, Scikit-learn, Matplotlib, Pytest, Streamlit Cloud.")
with cols[1]:
    with st.container(border=True):
        st.header("Limitations")
        st.write(
            "The bundled demo model is trained at low 64 x 64 resolution, so predictions depend heavily on dataset quality "
            "and may miss fine boundaries or confuse visually similar land classes. It is not suitable for official land "
            "surveys or policy decisions. Verified accuracy requires a labelled held-out evaluation dataset; larger datasets "
            "and stronger architectures can improve performance."
        )
with cols[2]:
    with st.container(border=True):
        st.header("GitHub Repository")
        st.markdown(f'<div class="footer-link"><a href="{REPO_URL}">View source code</a></div>', unsafe_allow_html=True)
        st.write("Source code, training scripts, evaluation pipeline, tests, and deployment configuration are included.")

with st.container(border=True):
    st.header("About Me")
    st.write("**Name:** Muhammad Aalim Baba")
    st.write("**Role:** Computer Science Engineering student / AI & Full-Stack developer")
    st.markdown(
        """
        <div class="legend">
          <div class="footer-link"><strong>GitHub:</strong> <a href="https://github.com/AalimBaba">github.com/AalimBaba</a></div>
          <div class="footer-link"><strong>LinkedIn:</strong> <a href="https://www.linkedin.com/in/aalimbaba-/">linkedin.com/in/aalimbaba-</a></div>
          <div><strong>Portfolio:</strong> Coming soon</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
