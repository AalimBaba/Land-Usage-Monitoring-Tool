from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from PIL import Image

from land_usage.config import CLASS_NAMES, PALETTE, REPO_URL
from land_usage.feedback import append_feedback, feedback_record, records_to_jsonl
from land_usage.inference import ModelUnavailableError, load_model, predict_image
from land_usage.uploads import UploadImageError, open_uploaded_image
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


def css(theme: str) -> None:
    if theme == "Dark":
        theme_vars = """
          --ink: #f8fafc;
          --muted: #b8c2cc;
          --line: rgba(148, 163, 184, .28);
          --field: rgba(20, 24, 31, .96);
          --panel: rgba(12, 15, 21, .78);
          --panel-strong: rgba(22, 27, 36, .92);
          --page-soft: #05070b;
          --page-base: #000000;
          --accent: #60a5fa;
          --accent-2: #22d3ee;
          --accent-3: #a78bfa;
          --warn: #f5b85b;
          --shadow: 0 22px 70px rgba(0,0,0,.54);
          --upload: rgba(18, 24, 35, .92);
          --success-bg: rgba(16, 185, 129, .12);
          --error-bg: rgba(244, 63, 94, .12);
          --info-bg: rgba(96, 165, 250, .12);
        """
        page_background = """
          radial-gradient(circle at 18% 6%, rgba(96, 165, 250, .20), transparent 32%),
          radial-gradient(circle at 78% 0%, rgba(167, 139, 250, .18), transparent 30%),
          linear-gradient(180deg, #05070b 0%, #0a0d14 46%, #000000 100%)
        """
    else:
        theme_vars = """
          --ink: #111827;
          --muted: #5f6b7a;
          --line: rgba(148, 163, 184, .36);
          --field: rgba(255, 255, 255, .94);
          --panel: rgba(255, 255, 255, .72);
          --panel-strong: rgba(255, 255, 255, .92);
          --page-soft: #f6f7fb;
          --page-base: #ffffff;
          --accent: #2563eb;
          --accent-2: #0891b2;
          --accent-3: #7c3aed;
          --warn: #a45f19;
          --shadow: 0 20px 55px rgba(15, 23, 42, .10);
          --upload: rgba(248, 250, 252, .96);
          --success-bg: rgba(16, 185, 129, .10);
          --error-bg: rgba(244, 63, 94, .10);
          --info-bg: rgba(37, 99, 235, .10);
        """
        page_background = """
          radial-gradient(circle at 12% 2%, rgba(96, 165, 250, .18), transparent 28%),
          radial-gradient(circle at 85% 4%, rgba(167, 139, 250, .16), transparent 30%),
          linear-gradient(180deg, #f8fafc 0%, #ffffff 48%, #f3f6fb 100%)
        """
    css_text = """
        <style>
        :root {
        __THEME_VARS__
          --glass: var(--panel);
          --glass-strong: var(--panel-strong);
        }
        html, body, .stApp, [data-testid="stAppViewContainer"] {
          color: var(--ink) !important;
          background: __PAGE_BACKGROUND__ !important;
          background-attachment: fixed !important;
        }
        [data-testid="stHeader"] {
          background: color-mix(in srgb, var(--panel-strong) 76%, transparent) !important;
          backdrop-filter: blur(20px) saturate(1.25);
          -webkit-backdrop-filter: blur(20px) saturate(1.25);
          border-bottom: 1px solid var(--line);
        }
        .block-container {
          padding-top: 1.4rem;
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
        .theme-bar {
          background: var(--glass);
          border: 1px solid var(--line);
          box-shadow: var(--shadow);
          backdrop-filter: blur(26px) saturate(1.28);
          -webkit-backdrop-filter: blur(26px) saturate(1.28);
          border-radius: 8px;
          padding: .85rem 1rem;
          margin-bottom: 1rem;
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
          color: var(--muted);
          font-size: clamp(1rem, 2vw, 1.25rem);
          line-height: 1.7;
        }
        .pill-row { display: flex; flex-wrap: wrap; gap: .65rem; margin-top: 1.3rem; }
        .pill {
          border: 1px solid var(--line);
          background: linear-gradient(135deg, color-mix(in srgb, var(--accent) 13%, var(--panel-strong)), var(--panel-strong));
          border-radius: 999px;
          padding: .55rem .78rem;
          color: var(--ink);
          font-weight: 700;
          font-size: .86rem;
        }
        .metric-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: .8rem;
        }
        .metric {
          background: var(--glass-strong);
          border: 1px solid var(--line);
          backdrop-filter: blur(18px) saturate(1.18);
          -webkit-backdrop-filter: blur(18px) saturate(1.18);
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
        .legend-item { display: flex; align-items: center; gap: .5rem; font-size: .9rem; color: var(--ink); }
        .swatch { width: 14px; height: 14px; border-radius: 3px; border: 1px solid rgba(0,0,0,.12); }
        .note {
          border-left: 4px solid var(--warn);
          background: color-mix(in srgb, var(--warn) 13%, var(--panel));
          padding: .9rem 1rem;
          border-radius: 6px;
          color: var(--ink);
        }
        .footer-link a { color: var(--accent-2); font-weight: 800; text-decoration: none; }
        [data-testid="stVerticalBlockBorderWrapper"] {
          background: var(--glass) !important;
          border-color: var(--line) !important;
          box-shadow: var(--shadow);
          backdrop-filter: blur(28px) saturate(1.22);
          -webkit-backdrop-filter: blur(28px) saturate(1.22);
        }
        [data-testid="stFileUploaderDropzone"] {
          background: var(--upload) !important;
          border: 1px dashed color-mix(in srgb, var(--accent) 62%, var(--line)) !important;
          border-radius: 10px;
        }
        [data-testid="stImage"] img {
          border-radius: 8px;
          border: 1px solid var(--line);
          background: var(--field);
        }
        div[data-testid="stMetric"] {
          background: var(--field) !important;
          border: 1px solid var(--line) !important;
          border-radius: 8px;
          padding: .7rem .8rem;
        }
        div[data-testid="stAlert"] {
          background: var(--glass-strong) !important;
          color: var(--ink) !important;
          border: 1px solid var(--line) !important;
          border-radius: 8px !important;
        }
        .stButton > button, .stDownloadButton > button {
          border-radius: 8px !important;
          border: 1px solid var(--line) !important;
          background: linear-gradient(135deg, color-mix(in srgb, var(--accent) 16%, var(--panel-strong)), var(--panel-strong)) !important;
          color: var(--ink) !important;
          box-shadow: 0 8px 20px color-mix(in srgb, var(--accent) 16%, transparent);
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
          border-color: var(--accent) !important;
          color: var(--accent-2) !important;
        }
        .stProgress > div > div {
          background-color: color-mix(in srgb, var(--muted) 14%, transparent) !important;
        }
        .stProgress div div div div {
          background: linear-gradient(90deg, var(--accent), var(--accent-2), var(--accent-3)) !important;
        }
        [data-testid="stSidebar"] {
          background: var(--panel-strong) !important;
          border-right: 1px solid var(--line);
        }
        [data-testid="stForm"] {
          background: color-mix(in srgb, var(--panel-strong) 78%, transparent) !important;
          border: 1px solid var(--line) !important;
          border-radius: 8px !important;
          padding: 1rem !important;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] textarea,
        textarea,
        input {
          background: var(--field) !important;
          color: var(--ink) !important;
          border-color: var(--line) !important;
          -webkit-text-fill-color: var(--ink) !important;
        }
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div,
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] * {
          color: var(--ink) !important;
          background-color: var(--field) !important;
        }
        div[data-baseweb="select"] svg,
        div[data-baseweb="checkbox"] svg,
        div[data-baseweb="radio"] svg {
          color: var(--ink) !important;
          fill: var(--ink) !important;
        }
        div[role="listbox"],
        div[role="option"] {
          background: var(--field) !important;
          color: var(--ink) !important;
        }
        div[role="option"] *,
        [data-baseweb="menu"] * {
          color: var(--ink) !important;
          background-color: var(--field) !important;
        }
        div[role="option"]:hover {
          background: color-mix(in srgb, var(--accent) 18%, var(--field)) !important;
        }
        [data-testid="stRadio"] label,
        [data-testid="stRadio"] p,
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzone"] small {
          color: var(--ink) !important;
        }
        h1, h2, h3, p, label, span, li, code { color: var(--ink); }
        [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] * {
          color: var(--ink);
        }
        small, .stCaptionContainer, [data-testid="stCaptionContainer"] {
          color: var(--muted) !important;
        }
        a { color: var(--accent-2) !important; }
        @media (max-width: 700px) {
          .block-container { padding-left: 1rem; padding-right: 1rem; }
          .hero { min-height: 340px; padding-top: 2rem; }
        }
        </style>
        """
    css_text = css_text.replace("__THEME_VARS__", theme_vars).replace("__PAGE_BACKGROUND__", page_background)
    st.markdown(
        css_text,
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


st.markdown('<div class="theme-bar">', unsafe_allow_html=True)
theme = st.radio("Theme", ["Light", "Dark"], horizontal=True, key="theme_toggle")
st.markdown("</div>", unsafe_allow_html=True)
css(theme)
metrics = load_metrics()
if "feedback_records" not in st.session_state:
    st.session_state.feedback_records = []


def render_feedback_panel(
    *,
    image: Image.Image,
    uploaded_filename: str | None,
    validation_status: str,
    segmentation_status: str,
    dominant: str | None = None,
    distribution: dict[str, float] | None = None,
    confidence: float | None = None,
    key: str,
) -> None:
    with st.container(border=True):
        st.subheader("Feedback")
        st.caption(
            "Privacy note: uploaded images are not stored by default. Feedback saves metadata only. "
            "Feedback is stored for this session only unless downloaded."
        )
        with st.form(f"feedback_form_{key}"):
            category = st.radio(
                "Was this result correct?",
                [
                    "Correct",
                    "Wrong: valid land image was rejected",
                    "Wrong: non-land image was accepted",
                    "Wrong: land class prediction looks incorrect",
                    "Other issue",
                ],
                key=f"feedback_category_{key}",
            )
            expected_type = st.selectbox(
                "Expected input type",
                [
                    "Satellite / aerial land image",
                    "Indoor photo",
                    "Document / ID / certificate",
                    "Screenshot",
                    "Portrait / selfie",
                    "Other",
                ],
                key=f"expected_type_{key}",
            )
            note = st.text_area("Describe the issue", key=f"feedback_note_{key}")
            submitted = st.form_submit_button("Submit feedback")

        if submitted:
            record = feedback_record(
                uploaded_filename=uploaded_filename,
                image=image,
                validation_status=validation_status,
                segmentation_status=segmentation_status,
                dominant_class=dominant,
                pixel_distribution=distribution,
                mean_softmax_confidence=confidence,
                feedback_category=category,
                user_note=note,
                expected_input_type=expected_type,
            )
            st.session_state.feedback_records.append(record)
            saved = append_feedback(record)
            if saved:
                st.success(
                    "Feedback added to this session log. Streamlit Cloud file storage is temporary, "
                    "so download the log if you want to keep it."
                )
            else:
                st.warning("Feedback kept in this session. Download the log if you want to keep it.")

        if st.session_state.feedback_records:
            st.download_button(
                "Download feedback log",
                data=records_to_jsonl(st.session_state.feedback_records),
                file_name="feedback_log.jsonl",
                mime="application/jsonl",
                key=f"download_feedback_{key}",
            )

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
            image = open_uploaded_image(upload)
            validation = validate_land_image(image)

            should_run_segmentation = validation.is_suitable
            if validation.is_rejected:
                render_validation_status(validation)
                st.error(
                    "This image does not appear to be satellite or aerial land imagery. "
                    "Please upload a suitable land or satellite image."
                )
                render_feedback_panel(
                    image=image,
                    uploaded_filename=upload.name,
                    validation_status=validation.image_relevance,
                    segmentation_status="Not run",
                    key="rejected",
                )
                should_run_segmentation = False
            elif validation.is_uncertain:
                render_validation_status(validation)
                st.warning(
                    "The image is ambiguous. Segmentation is disabled by default, but you may run it manually for demonstration."
                )
                should_run_segmentation = st.button("Run segmentation anyway", type="secondary")
                if not should_run_segmentation:
                    render_feedback_panel(
                        image=image,
                        uploaded_filename=upload.name,
                        validation_status=validation.image_relevance,
                        segmentation_status="Not run",
                        key="uncertain",
                    )

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
            render_feedback_panel(
                image=image,
                uploaded_filename=upload.name,
                validation_status=validation.image_relevance,
                segmentation_status="Run",
                dominant=dominant_name,
                distribution=result.class_distribution,
                confidence=result.mean_confidence,
                key="result",
            )
        except UploadImageError as exc:
            st.error(str(exc))
        except ModelUnavailableError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Prediction failed gracefully: {exc}")

overview_cols = st.columns(3)
with overview_cols[0]:
    with st.container(border=True):
        st.header("Problem Statement")
        st.write(
            "Land-use monitoring needs quick interpretation of satellite or aerial imagery, but raw imagery is hard to scan manually. "
            "This demo shows how semantic segmentation can convert an uploaded land image into a visual land-cover mask."
        )
with overview_cols[1]:
    with st.container(border=True):
        st.header("Solution Overview")
        st.write(
            "The app validates whether an upload looks like land or aerial imagery, runs a compact TensorFlow/Keras U-Net when suitable, "
            "and displays the predicted mask, overlay, pixel distribution, and feedback controls."
        )
with overview_cols[2]:
    with st.container(border=True):
        st.header("Key Features")
        st.write(
            "Upload validation, real model inference, segmentation overlay, predicted pixel distribution, metadata-only feedback, "
            "downloadable feedback logs, and honest evaluation messaging are included."
        )

pipeline_cols = st.columns(2)
with pipeline_cols[0]:
    with st.container(border=True):
        st.header("Architecture / Pipeline")
        st.write(
            "Upload -> file decoder -> input relevance validator -> optional U-Net inference -> mask colorization -> overlay rendering -> "
            "class distribution -> feedback capture."
        )
        st.write(
            "Training utilities support aligned image/mask resizing, label remapping, train/validation/test splits, augmentation, "
            "checkpointing, and segmentation metrics."
        )
with pipeline_cols[1]:
    with st.container(border=True):
        st.header("Demo Workflow")
        st.write(
            "Use a satellite crop, aerial city image, farmland scene, coastline, vegetation image, or mixed land-cover image. "
            "Documents, ID cards, portraits, indoor rooms, blank images, and screenshots are blocked or marked uncertain before segmentation."
        )

with st.container(border=True):
    st.header("Accuracy & Evaluation")
    st.write(
        "Input-validation accuracy can be benchmarked with `python validation_benchmark.py`. "
        "That benchmark measures the upload gate only; it is not segmentation/model accuracy."
    )
    st.write(
        "Segmentation accuracy requires labelled satellite image and mask pairs. Supported model metrics include Pixel Accuracy, "
        "Mean IoU, Mean Dice, Macro F1, per-class IoU, precision, recall, F1-score, and a confusion matrix. "
        "This app does not fabricate accuracy values."
    )
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
            'Current deployed model accuracy cannot be verified without labelled satellite images and matching masks from a held-out test dataset. '
            'A 95%+ accuracy claim cannot be made honestly until it is measured on that dataset. '
            'Run <code>python evaluate_model.py --data-dir "path/to/SEN-2 LULC" --model unet_model.h5</code> '
            'with labelled image/mask pairs to generate <code>metrics.json</code>. Supported metrics include '
            'Pixel Accuracy, Mean IoU, Mean Dice, Macro F1, per-class IoU, precision, recall, F1-score, and a confusion matrix.</div>',
            unsafe_allow_html=True,
        )

detail_cols = st.columns(3)
with detail_cols[0]:
    with st.container(border=True):
        st.header("Limitations")
        st.write(
            "The bundled model is a compact demo trained at low 64 x 64 resolution. It can confuse visually similar classes, miss fine boundaries, "
            "and should not be used for official surveys, policy decisions, or safety-critical monitoring."
        )
with detail_cols[1]:
    with st.container(border=True):
        st.header("How to Improve Accuracy")
        st.write(
            "Collect labelled satellite images and masks, create a train/validation/test split, retrain above 64 x 64 resolution, "
            "try U-Net++, DeepLabV3+, or SegFormer, measure Pixel Accuracy, Mean IoU, Dice, and Macro F1, and update `metrics.json` "
            "only with real held-out results."
        )
with detail_cols[2]:
    with st.container(border=True):
        st.header("Tech Stack")
        st.write("Python, Streamlit, TensorFlow/Keras, U-Net, NumPy, Pillow, Scikit-learn, Matplotlib, Pytest, Streamlit Cloud.")

with st.container(border=True):
    st.header("About Me")
    st.write("**Name:** Muhammad Aalim Baba")
    st.write("**Role:** Computer Science Engineering student | AI/ML & Full-Stack Developer")
    st.markdown(
        """
        <div class="legend">
          <div class="footer-link"><strong>GitHub:</strong> <a href="https://github.com/AalimBaba">github.com/AalimBaba</a></div>
          <div class="footer-link"><strong>LinkedIn:</strong> <a href="https://www.linkedin.com/in/aalimbaba-/">linkedin.com/in/aalimbaba-</a></div>
          <div class="footer-link"><strong>Portfolio:</strong> <a href="https://aalimbaba.github.io/Portfolio/">aalimbaba.github.io/Portfolio</a></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.container(border=True):
    st.header("Links")
    st.markdown(
        f"""
        <div class="legend">
          <div class="footer-link"><strong>Live Demo:</strong> <a href="https://land-usage-monitoring-tool.streamlit.app/">land-usage-monitoring-tool.streamlit.app</a></div>
          <div class="footer-link"><strong>GitHub Repository:</strong> <a href="{REPO_URL}">AalimBaba/Land-Usage-Monitoring-Tool</a></div>
          <div class="footer-link"><strong>Validation Benchmark:</strong> run <code>python validation_benchmark.py</code></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
