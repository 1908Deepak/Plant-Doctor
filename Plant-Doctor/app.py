"""
Streamlit application for Plant Doctor — Plant Disease Prediction.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from main import Predictor, Config, PredictorError


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="🌿 Plant Doctor — Streamlit",
    page_icon="🌿",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Project metadata (edit these to your project)
# -----------------------------------------------------------------------------
PROJECT = {
    "name": "Plant Doctor",
    "tagline": "AI-powered plant disease detection",
    "version": "v1.0.0",
    "dataset": "https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/b4e3a32f-c0bd-4060-81e9-6144231f2520/file_downloaded",
    "algorithm": "TensorFlow/Keras CNN (transfer learning)",
    "license": "MIT",
    "repo": "https://github.com/1908Deepak/Plant-Doctor",
}
AUTHOR = {
    "name": "Deepak Singh",
    "role": "ML Engineer / Developer",
    "email": "deepaksingh190810@gmail.com",
    "linkedin": "https://www.linkedin.com/in/1908Deepak/",
    "github": "https://github.com/1908Deepak",
    "Portfolio": "https://1908deepak.vercel.app/"
}


# -----------------------------------------------------------------------------
# Caching layer
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_predictor() -> Predictor:
    """Cache the Predictor instance across reruns."""
    p = Predictor(Config())
    p.load()
    return p


@st.cache_data(show_spinner=False)
def get_labels():
    try:
        return get_predictor().labels
    except Exception:
        return []


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _predict_image(img: Image.Image) -> Dict:
    """Save the PIL image to temp file and run prediction."""
    predictor = get_predictor()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name, format="PNG")
        res = predictor.predict(Path(tmp.name))
    return res


def _valid_image(file_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return img
    except UnidentifiedImageError as exc:
        raise PredictorError("Unsupported or corrupted image file.") from exc


# -----------------------------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("assets/logo.svg", width=64)
    st.title(PROJECT["name"])
    page = st.radio("Navigate", ["Detect", "Project Details", "Author"], index=0)
    st.caption(f"Version {PROJECT['version']}")


# -----------------------------------------------------------------------------
# Detect page
# -----------------------------------------------------------------------------
if page == "Detect":
    st.title(f"{PROJECT['name']} — {PROJECT['tagline']}")
    st.write("Upload a clear photo of a single leaf. Supported formats: JPG/PNG/WebP.")

    # Show the model's expected input size (H×W×C)
    _pred = get_predictor()
    _in_size = _pred.input_size
    if _in_size:
        st.caption(f"Model expects **{_in_size[0]}×{_in_size[1]}×{_in_size[2]}**. "
                   "Uploaded images will be resized automatically.")

    uploaded = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png", "webp"])
    col1, col2 = st.columns([1, 1])

    if uploaded is not None:
        try:
            img = _valid_image(uploaded.read())
        except PredictorError as e:
            st.error(str(e))
            st.stop()

        with col1:
            st.subheader("Preview")
            st.image(img, caption="Uploaded image", use_column_width=True)

        with col2:
            st.subheader("Prediction")
            with st.spinner("Analyzing…"):
                result = _predict_image(img)
            st.success(f"**{result['label']}**  •  confidence: **{result['confidence']*100:.2f}%**")

            info = result.get('info') or {}
            if info:
                with st.expander('Cause & Cure', expanded=True):
                    st.markdown(f"**Cause:** {info.get('cause', '—')}")
                    st.markdown(f"**Recommended Action:** {info.get('cure', '—')}")

            if result.get("probs"):
                probs = pd.DataFrame({
                    "class": list(result["probs"].keys()),
                    "probability": list(result["probs"].values())
                }).sort_values("probability", ascending=False)
                st.write("Top probabilities:")
                st.dataframe(probs, hide_index=True, use_container_width=True)
                st.bar_chart(probs.set_index("class"))

    st.info("Tip: best results come from sharp, well-lit images with a single leaf filling most of the frame.")


# -----------------------------------------------------------------------------
# Project Details page
# -----------------------------------------------------------------------------
elif page == "Project Details":
    st.title("Project Details")
    left, right = st.columns([1, 1])

    with left:
        st.header("Overview")
        st.write(f"**Project:** {PROJECT['name']}  \n**Version:** {PROJECT['version']}  \n**License:** {PROJECT['license']}")
        st.write(f"**Tagline:** {PROJECT['tagline']}")
        
        st.write(f"**Dataset:** {PROJECT['dataset']}")
        st.write(f"**Algorithm:** {PROJECT['algorithm']}")

    with right:
        st.header("Tech Stack")
        st.write("- Streamlit for UI")
        st.write("- TensorFlow/Keras for modeling")
        st.write("- Pillow for image I/O")
        st.write("- NumPy & Pandas for utilities")
        if PROJECT.get("repo"):
            st.write(f"**Repository:** {PROJECT['repo']}")

    st.header("How it works")
    st.markdown(
        """
        1. The uploaded image is resized to the **model's expected size** (e.g., 160×160) and normalized to **[0, 1]**.
        2. The Keras model predicts class probabilities.
        3. The app displays the **top class** and a **bar chart** of probabilities.
        """
    )

    st.header("Classes")
    labels = get_labels()
    if labels:
        st.write(", ".join(labels))
    else:
        st.caption("Labels not available — ensure `labels.json` exists.")


# -----------------------------------------------------------------------------
# Author page
# -----------------------------------------------------------------------------
else:
    st.title("Author")
    st.subheader(AUTHOR["name"])
    st.write(AUTHOR["role"])
    st.write(f"Email: {AUTHOR['email']}")
    st.write(f"[LinkedIn]({AUTHOR['linkedin']})  |  [GitHub]({AUTHOR['github']})  |  [Portfolio]({AUTHOR['Portfolio']})")
    st.caption("Update these fields in `app.py`.")
