import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import json
import os
import base64
import textwrap
from io import BytesIO


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NeuroScan AI | Brain Tumor MRI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS  (only these two files are required)
# ============================================================

MODEL_PATH = "best_brain_tumor_model.h5"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = 224

DEFAULT_CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

DISPLAY_MODEL_NAME = "MobileNetV2"
DISPLAY_MODEL_TYPE = "Fine-tuned"


# ============================================================
# HTML RENDERER  (prevents raw HTML / code-block bug)
# ============================================================

def render_html(html: str) -> None:
    """Strip leading spaces so Streamlit does not treat HTML as a code block."""
    cleaned = textwrap.dedent(str(html)).strip()
    cleaned = "\n".join(line.lstrip() for line in cleaned.splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


# ============================================================
# CSS
# ============================================================

render_html("""
<style>
.stApp {
background: radial-gradient(circle at 85% 5%, rgba(32,91,190,0.12), transparent 30%),
linear-gradient(135deg, #050D19 0%, #071426 45%, #08172B 100%);
color: #F4F7FF;
}
.block-container {
max-width: 1500px;
/* Extra top space so Streamlit toolbar does not clip the title card border */
padding-top: 3.2rem !important;
padding-bottom: 2rem;
padding-left: 1.4rem;
padding-right: 1.4rem;
}
/* Keep Streamlit header visible so:
   - sidebar can collapse / expand with <<
   - Deploy / Settings / menu show top-right
*/
footer { visibility: hidden; }

header[data-testid="stHeader"] {
background: rgba(5, 13, 25, 0.88) !important;
backdrop-filter: blur(10px);
}

/* Sidebar collapse / expand control always visible */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
button[kind="headerNoPadding"],
button[kind="header"] {
visibility: visible !important;
display: flex !important;
color: #AFC3E3 !important;
opacity: 1 !important;
}

/* Top-right toolbar: Deploy, menu, etc. */
#MainMenu { visibility: visible !important; }
header[data-testid="stHeader"] button,
header[data-testid="stHeader"] [data-testid="stToolbar"] button,
header[data-testid="stHeader"] a {
color: #AFC3E3 !important;
visibility: visible !important;
opacity: 1 !important;
}
header[data-testid="stHeader"] button:hover {
color: #FFFFFF !important;
}

section[data-testid="stSidebar"] {
background: linear-gradient(180deg, #071326 0%, #08172C 55%, #061120 100%);
border-right: 1px solid #1A3151;
}
section[data-testid="stSidebar"] > div { padding-top: 0.6rem; }
section[data-testid="stSidebar"] * { color: #DCE8FA; }

/* ---- Nav icon buttons ---- */
section[data-testid="stSidebar"] div.stButton > button {
width: 100% !important;
background: transparent !important;
border: none !important;
border-radius: 9px !important;
padding: 11px 14px !important;
margin-bottom: 5px !important;
color: #AFC3E3 !important;
font-size: 0.92rem !important;
font-weight: 500 !important;
text-align: left !important;
justify-content: flex-start !important;
box-shadow: none !important;
}
section[data-testid="stSidebar"] div.stButton > button:hover {
background: rgba(18, 61, 125, 0.45) !important;
color: #FFFFFF !important;
border: none !important;
}
section[data-testid="stSidebar"] div.stButton > button[kind="primary"],
section[data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-primary"] {
background: linear-gradient(90deg, #123D7D, #12346B) !important;
color: #FFFFFF !important;
box-shadow: inset 0 0 0 1px rgba(58, 142, 255, 0.12) !important;
}

.sidebar-brand { text-align: center; padding: 8px 5px 14px 5px; }
.sidebar-brain {
font-size: 48px; line-height: 1; margin-bottom: 6px;
filter: drop-shadow(0 0 15px rgba(50,137,255,0.75));
}
.sidebar-logo { font-size: 1.45rem; font-weight: 800; letter-spacing: -0.6px; color: #FFFFFF; }
.sidebar-logo span { color: #765BFF; }
.sidebar-subtitle { color: #8FA7C9; font-size: 0.76rem; margin-top: 4px; }
.sidebar-divider { height: 1px; background: #263A57; margin: 10px 0 14px 0; }

.sidebar-section {
color: #7189AE; font-size: 0.68rem; font-weight: 800;
letter-spacing: 1.2px; text-transform: uppercase;
margin-top: 18px; margin-bottom: 10px;
}
.class-row {
display: flex; align-items: center; gap: 12px;
margin: 10px 0; color: #CBD8EC; font-size: 0.88rem;
}
.class-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
.glioma { background: #EF476F; box-shadow: 0 0 8px rgba(239,71,111,0.4); }
.meningioma { background: #2196F3; box-shadow: 0 0 8px rgba(33,150,243,0.4); }
.no-tumor { background: #4FD39A; box-shadow: 0 0 8px rgba(79,211,154,0.4); }
.pituitary { background: #A855F7; box-shadow: 0 0 8px rgba(168,85,247,0.4); }

.model-detail { display: flex; gap: 12px; margin: 12px 0; }
.detail-icon { width: 24px; font-size: 1rem; color: #6FB8FF; }
.detail-title { color: #E8F0FD; font-size: 0.85rem; }
.detail-value { color: #8DA5C7; font-size: 0.75rem; margin-top: 2px; }
.sidebar-disclaimer {
border-top: 1px solid #263A57; margin-top: 20px; padding-top: 14px;
color: #7389A8; font-size: 0.72rem; line-height: 1.5;
}

.top-header {
min-height: 120px;
border: 1px solid #286BDA;
border-radius: 18px;
background: linear-gradient(110deg, #09172C 0%, #102C61 55%, #123A70 100%);
box-shadow: 0 10px 35px rgba(0,0,0,0.30), inset 0 1px 0 rgba(255,255,255,0.04);
display: flex;
align-items: center;
padding: 22px 30px;
margin-top: 0.4rem;
margin-bottom: 16px;
position: relative;
overflow: hidden;
}
.top-header::after {
content: "🧠"; position: absolute; right: 24px; top: -12px;
font-size: 100px; opacity: 0.15; filter: drop-shadow(0 0 22px #2F8CFF);
}
.brand-mark {
width: 60px; height: 60px; border-radius: 16px; display: flex;
align-items: center; justify-content: center;
background: linear-gradient(135deg, #6235F4, #168DFF); font-size: 32px;
margin-right: 18px; box-shadow: 0 0 22px rgba(72,110,255,0.40);
position: relative; z-index: 2;
}
.header-text { position: relative; z-index: 2; }
.neuroscan-title { font-size: 2rem; font-weight: 800; letter-spacing: -1px; color: #FFFFFF; }
.neuroscan-title span { color: #765BFF; }
.neuroscan-subtitle { color: #AEC4E6; font-size: 0.98rem; margin-top: 2px; }
.header-model {
position: absolute; right: 130px; top: 26px; z-index: 3;
border-left: 1px solid #315177; padding-left: 24px;
}
.header-model-label { color: #8DA4C6; font-size: 0.74rem; }
.header-model-name { color: #FFFFFF; font-size: 1.12rem; font-weight: 750; margin-top: 3px; }
.header-model-type { color: #8DA4C6; font-size: 0.74rem; margin-top: 2px; }

.upload-heading { color: #FFFFFF; font-size: 1.2rem; font-weight: 750; text-align: center; margin-top: 4px; }
.upload-description { color: #849CC0; font-size: 0.86rem; text-align: center; margin-top: 4px; margin-bottom: 8px; }

[data-testid="stFileUploader"] {
background: linear-gradient(135deg, #09192F, #0B1B32) !important;
border: 1px dashed #198DFF !important; border-radius: 17px !important;
padding: 22px 28px 14px 28px !important;
box-shadow: inset 0 0 35px rgba(23,126,255,0.04); margin-bottom: 16px;
}
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"] { background: transparent !important; border: none !important; }
[data-testid="stFileUploaderDropzoneInstructions"] { color: #AFC3E3 !important; }
[data-testid="stFileUploaderDropzoneInstructions"] svg { color: #63AFFF !important; }
[data-testid="stFileUploader"] button {
background: linear-gradient(90deg, #167BFF, #7955F7) !important;
color: #FFFFFF !important; border: none !important; border-radius: 9px !important; font-weight: 700 !important;
}
[data-testid="stFileUploader"] small { color: #7891B5 !important; }

.empty-state {
background: #102B49; border-radius: 10px; padding: 14px 18px;
color: #258FFF; font-size: 0.92rem; margin-top: 4px;
}

.dashboard-card {
background: linear-gradient(145deg, #0A1930, #081528);
border: 1px solid #1767C5; border-radius: 17px; padding: 18px;
box-shadow: 0 8px 28px rgba(0,0,0,0.20); min-height: 380px;
}
.card-heading {
display: flex; align-items: center; gap: 10px;
color: #F5F8FF; font-size: 1.05rem; font-weight: 750; margin-bottom: 14px;
}
.card-heading-icon { color: #29AEFF; font-size: 1.3rem; }

.mri-frame {
width: 100%; max-width: 420px; height: 270px; margin: 0 auto;
border: 1px solid #247FD5; border-radius: 9px; overflow: hidden;
background: #000; display: flex; align-items: center; justify-content: center;
box-shadow: 0 0 18px rgba(23,126,255,0.10);
}
.mri-frame img { width: 100%; height: 100%; object-fit: contain; display: block; }
.image-info {
display: flex; background: linear-gradient(90deg, #102746, #10213A);
border-radius: 10px; margin-top: 12px; padding: 10px; gap: 28px;
}
.image-info-item { flex: 1; }
.image-info-value { color: #FFFFFF; font-weight: 700; font-size: 0.86rem; }
.image-info-label { color: #8CA4C7; font-size: 0.7rem; margin-top: 2px; }

.prediction-card {
background: linear-gradient(135deg, #35152C, #25142B);
border: 1px solid #E62E67; border-radius: 13px; padding: 16px;
box-shadow: 0 0 25px rgba(230,46,103,0.10); margin-bottom: 16px;
}
.prediction-label {
color: #BE92A8; font-size: 0.76rem; font-weight: 750;
text-transform: uppercase; letter-spacing: 0.8px;
}
.prediction-class { color: #FF527F; font-size: 1.85rem; font-weight: 800; margin-top: 2px; }
.prediction-confidence { color: #F4EAF0; font-size: 0.95rem; font-weight: 600; }
.probability-heading { color: #E7EEFB; font-size: 0.9rem; font-weight: 750; margin-bottom: 12px; }
.prob-row {
display: grid; grid-template-columns: 95px 1fr 60px;
gap: 10px; align-items: center; margin: 11px 0;
}
.prob-name { color: #D8E2F2; font-size: 0.82rem; }
.prob-track { height: 9px; background: #172941; border-radius: 99px; overflow: hidden; }
.prob-value { text-align: right; color: #DDE7F7; font-size: 0.78rem; font-weight: 650; }
.prob-bar { height: 100%; border-radius: 99px; min-width: 2px; }
.bar-glioma { background: #E43B67; }
.bar-meningioma { background: #2196F3; }
.bar-no-tumor { background: #4FD39A; }
.bar-pituitary { background: #A855F7; }

.model-info-card {
background: linear-gradient(145deg, #0A1930, #081528);
border: 1px solid #1767C5; border-radius: 17px; padding: 16px 22px; margin-top: 16px;
}
.model-info-title { color: #EAF1FD; font-size: 1rem; font-weight: 750; margin-bottom: 14px; }
.model-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0; }
.model-item { padding: 8px 20px; border-right: 1px solid #29415F; }
.model-item:first-child { padding-left: 0; }
.model-item:last-child { border-right: none; }
.model-item-label { color: #7E97BA; font-size: 0.72rem; }
.model-item-value { color: #F3F7FF; font-size: 0.95rem; font-weight: 750; margin-top: 3px; }
.model-item-sub { color: #8DA4C6; font-size: 0.72rem; margin-top: 2px; }

.disclaimer {
background: linear-gradient(90deg, #2D2616, #221D13);
border: 1px solid #80611C; border-radius: 13px; padding: 12px 16px; margin-top: 12px;
}
.disclaimer-title { color: #F5B93E; font-size: 0.8rem; font-weight: 750; }
.disclaimer-text { color: #C9B98E; font-size: 0.72rem; line-height: 1.45; margin-top: 3px; }
.footer { text-align: center; color: #526782; font-size: 0.72rem; margin-top: 16px; padding-bottom: 8px; }

.info-page-card {
background: linear-gradient(145deg, #0A1930, #081528);
border: 1px solid #1767C5; border-radius: 17px; padding: 28px 32px;
box-shadow: 0 8px 28px rgba(0,0,0,0.20); margin-top: 8px;
}
.info-page-card h2 { color: #F3F7FF; font-size: 1.45rem; margin: 0 0 12px 0; }
.info-page-card p, .info-page-card li {
color: #AEC4E6; font-size: 0.95rem; line-height: 1.65; margin: 0 0 10px 0;
}
.info-page-card ul { padding-left: 18px; margin: 8px 0 0 0; }
.info-stat-grid {
display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; margin-top: 18px;
}
.info-stat {
background: #0D1F38; border: 1px solid #1E4A7A; border-radius: 12px; padding: 14px 16px;
}
.info-stat-label { color: #7E97BA; font-size: 0.72rem; }
.info-stat-value { color: #F3F7FF; font-size: 1.05rem; font-weight: 750; margin-top: 4px; }

@media (max-width: 900px) {
.header-model { display: none; }
.neuroscan-title { font-size: 1.55rem; }
.top-header::after { display: none; }
.model-grid { grid-template-columns: 1fr; }
.model-item { border-right: none; border-bottom: 1px solid #29415F; padding: 12px 0; }
.model-item:last-child { border-bottom: none; }
.prob-row { grid-template-columns: 80px 1fr 55px; }
.mri-frame { height: 240px; }
.info-stat-grid { grid-template-columns: 1fr; }
}
</style>
""")


# ============================================================
# LOAD MODEL + CLASS NAMES
# ============================================================

@st.cache_resource
def load_brain_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return load_model(MODEL_PATH, compile=False)


@st.cache_data
def load_class_names():
    if os.path.exists(CLASS_NAMES_PATH):
        try:
            with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) >= 4:
                return data[:4]
            if isinstance(data, dict) and len(data) >= 4:
                idx_to_name = {int(v): k for k, v in data.items()}
                return [idx_to_name[i] for i in sorted(idx_to_name.keys())]
        except Exception:
            pass
    return DEFAULT_CLASS_NAMES


def preprocess_image(image):
    """Keep the same preprocessing you confirmed works with your model."""
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    arr = np.array(image, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def predict(model, image, class_names):
    arr = preprocess_image(image)
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return class_names[idx], float(preds[idx]) * 100.0, preds


def image_to_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ============================================================
# SESSION STATE (for icon navigation)
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"


def set_page(name: str) -> None:
    st.session_state.page = name


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    render_html("""
<div class="sidebar-brand">
<div class="sidebar-brain">🧠</div>
<div class="sidebar-logo">NeuroScan <span>AI</span></div>
<div class="sidebar-subtitle">Brain Tumor MRI Classification</div>
</div>
<div class="sidebar-divider"></div>
""")

    # Icon navigation buttons (active page uses primary type)
    st.button(
        "🏠  Home",
        key="nav_home",
        use_container_width=True,
        type="primary" if st.session_state.page == "Home" else "secondary",
        on_click=set_page,
        args=("Home",),
    )
    st.button(
        "ℹ️  About",
        key="nav_about",
        use_container_width=True,
        type="primary" if st.session_state.page == "About" else "secondary",
        on_click=set_page,
        args=("About",),
    )
    st.button(
        "⚙️  Model Info",
        key="nav_model",
        use_container_width=True,
        type="primary" if st.session_state.page == "Model Info" else "secondary",
        on_click=set_page,
        args=("Model Info",),
    )

    render_html("""
<div class="sidebar-divider"></div>
<div class="sidebar-section">Classes</div>
<div class="class-row"><div class="class-dot glioma"></div><div>Glioma</div></div>
<div class="class-row"><div class="class-dot meningioma"></div><div>Meningioma</div></div>
<div class="class-row"><div class="class-dot no-tumor"></div><div>No Tumor</div></div>
<div class="class-row"><div class="class-dot pituitary"></div><div>Pituitary</div></div>
<div class="sidebar-section">Model Details</div>
<div class="model-detail"><div class="detail-icon">▣</div><div><div class="detail-title">MobileNetV2</div><div class="detail-value">(Fine-tuned)</div></div></div>
<div class="model-detail"><div class="detail-icon">▧</div><div><div class="detail-title">Input Size</div><div class="detail-value">224 × 224</div></div></div>
<div class="model-detail"><div class="detail-icon">⚙</div><div><div class="detail-title">Framework</div><div class="detail-value">TensorFlow / Keras</div></div></div>
<div class="model-detail"><div class="detail-icon">⌘</div><div><div class="detail-title">Output</div><div class="detail-value">4 Classes</div></div></div>
<div class="sidebar-disclaimer">🛡️ For educational and research purposes only. Not a substitute for professional medical diagnosis.</div>
""")

page = st.session_state.page


# ============================================================
# SHARED HEADER
# ============================================================

render_html(f"""
<div class="top-header">
<div class="brand-mark">🧠</div>
<div class="header-text">
<div class="neuroscan-title">NeuroScan <span>AI</span></div>
<div class="neuroscan-subtitle">Brain Tumor MRI Classification</div>
</div>
<div class="header-model">
<div class="header-model-label">Model</div>
<div class="header-model-name">{DISPLAY_MODEL_NAME}</div>
<div class="header-model-type">{DISPLAY_MODEL_TYPE}</div>
</div>
</div>
""")


# ============================================================
# PAGE: ABOUT
# ============================================================

if page == "About":
    render_html("""
<div class="info-page-card">
<h2>About NeuroScan AI</h2>
<p>
NeuroScan AI is a deep-learning application that classifies brain MRI scans into four categories:
<strong>Glioma</strong>, <strong>Meningioma</strong>, <strong>Pituitary</strong>, and <strong>No Tumor</strong>.
</p>
<p>
It is designed to assist students, researchers, and clinicians in exploring AI-assisted medical imaging.
Predictions are generated instantly after you upload an MRI image.
</p>
<ul>
<li>Upload JPG / JPEG / PNG brain MRI images</li>
<li>Get class prediction with confidence score</li>
<li>View probability distribution across all four classes</li>
<li>Built with TensorFlow / Keras and Streamlit</li>
</ul>
<p style="margin-top:16px;color:#F5B93E;">
⚠ This tool is for educational and research use only. It is not a medical diagnosis system.
</p>
</div>
<div class="footer">NeuroScan AI · Brain Tumor MRI Classification &nbsp;|&nbsp; Built with TensorFlow + Streamlit</div>
""")
    st.stop()


# ============================================================
# PAGE: MODEL INFO
# ============================================================

if page == "Model Info":
    render_html(f"""
<div class="info-page-card">
<h2>Model Information</h2>
<p>
The deployed model is a fine-tuned <strong>{DISPLAY_MODEL_NAME}</strong> network trained on labeled brain MRI images
for multi-class tumor classification.
</p>
<div class="info-stat-grid">
<div class="info-stat"><div class="info-stat-label">Architecture</div><div class="info-stat-value">{DISPLAY_MODEL_NAME}</div></div>
<div class="info-stat"><div class="info-stat-label">Training Mode</div><div class="info-stat-value">{DISPLAY_MODEL_TYPE}</div></div>
<div class="info-stat"><div class="info-stat-label">Input Size</div><div class="info-stat-value">224 × 224 × 3</div></div>
<div class="info-stat"><div class="info-stat-label">Framework</div><div class="info-stat-value">TensorFlow / Keras</div></div>
<div class="info-stat"><div class="info-stat-label">Output Classes</div><div class="info-stat-value">4 Classes</div></div>
<div class="info-stat"><div class="info-stat-label">Model File</div><div class="info-stat-value">best_brain_tumor_model.h5</div></div>
</div>
<p style="margin-top:18px;">
<strong>Classes:</strong> Glioma · Meningioma · No Tumor · Pituitary
</p>
<p style="color:#F5B93E;">
⚠ Predictions are probabilistic estimates and must not replace clinical evaluation by a qualified professional.
</p>
</div>
<div class="footer">NeuroScan AI · Brain Tumor MRI Classification &nbsp;|&nbsp; Built with TensorFlow + Streamlit</div>
""")
    st.stop()


# ============================================================
# PAGE: HOME  (upload + predict)
# ============================================================

try:
    model = load_brain_model()
except Exception as e:
    st.error("Unable to load the trained model.")
    st.exception(e)
    st.stop()

class_names = load_class_names()

if model is None:
    st.error(
        f"**Model file not found.**\n\n"
        f"Place `{MODEL_PATH}` in the same folder as `app.py`.\n\n"
        f"Also place `{CLASS_NAMES_PATH}` next to it."
    )
    st.stop()

render_html("""
<div class="upload-heading">Upload your MRI scan</div>
<div class="upload-description">Drag & drop an image here, or browse files</div>
""")

uploaded_file = st.file_uploader(
    "Browse MRI files",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    render_html("""
<div class="empty-state">👋 Upload a brain MRI image to get started.</div>
<div class="footer">NeuroScan AI · Brain Tumor MRI Classification &nbsp;|&nbsp; Built with TensorFlow + Streamlit</div>
""")
    st.stop()

try:
    image = Image.open(uploaded_file).convert("RGB")
except Exception:
    st.error("Unable to read the uploaded image.")
    st.stop()

with st.spinner("Analyzing MRI scan..."):
    pred_class, confidence, all_preds = predict(model, image, class_names)

image_b64 = image_to_base64(image)

bar_map = {
    "Glioma": "bar-glioma",
    "Meningioma": "bar-meningioma",
    "No Tumor": "bar-no-tumor",
    "Pituitary": "bar-pituitary",
}

prob_html = ""
for name, prob in zip(class_names, all_preds):
    pct = float(prob) * 100.0
    width = min(max(pct, 0.0), 100.0)
    bar = bar_map.get(name, "bar-meningioma")
    prob_html += (
        f'<div class="prob-row">'
        f'<div class="prob-name">{name}</div>'
        f'<div class="prob-track"><div class="prob-bar {bar}" style="width:{width:.4f}%"></div></div>'
        f'<div class="prob-value">{pct:.2f}%</div>'
        f"</div>"
    )

dashboard = f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:22px;margin-top:18px;">
<div class="dashboard-card">
<div class="card-heading"><div class="card-heading-icon">▧</div><div>MRI PREVIEW</div></div>
<div class="mri-frame"><img src="data:image/png;base64,{image_b64}" alt="Uploaded MRI"></div>
<div class="image-info">
<div class="image-info-item"><div class="image-info-value">{image.width} × {image.height}</div><div class="image-info-label">Image size</div></div>
<div style="width:1px;background:#29415F;"></div>
<div class="image-info-item"><div class="image-info-value">RGB</div><div class="image-info-label">Color mode</div></div>
</div>
</div>
<div class="dashboard-card">
<div class="card-heading"><div class="card-heading-icon" style="color:#A855F7;">▥</div><div>AI ANALYSIS</div></div>
<div class="prediction-card">
<div class="prediction-label">◉ &nbsp; PREDICTION</div>
<div class="prediction-class">{pred_class}</div>
<div class="prediction-confidence">{confidence:.2f}% confidence</div>
</div>
<div class="probability-heading">▥ &nbsp; CLASS PROBABILITIES</div>
{prob_html}
</div>
</div>
<div class="model-info-card">
<div class="model-info-title">⚙ &nbsp; MODEL INFORMATION</div>
<div class="model-grid">
<div class="model-item"><div class="model-item-label">Architecture</div><div class="model-item-value">{DISPLAY_MODEL_NAME}</div><div class="model-item-sub">{DISPLAY_MODEL_TYPE}</div></div>
<div class="model-item"><div class="model-item-label">Input Size</div><div class="model-item-value">224 × 224</div><div class="model-item-sub">RGB image input</div></div>
<div class="model-item"><div class="model-item-label">Number of Classes</div><div class="model-item-value">4 Classes</div><div class="model-item-sub">Glioma, Meningioma, No Tumor, Pituitary</div></div>
</div>
</div>
<div class="disclaimer">
<div class="disclaimer-title">⚠ &nbsp; Important Notice</div>
<div class="disclaimer-text">This application is intended for educational and research purposes only. The prediction should not be considered a medical diagnosis or a replacement for evaluation by a qualified healthcare professional.</div>
</div>
<div class="footer">NeuroScan AI · Brain Tumor MRI Classification &nbsp;|&nbsp; Built with TensorFlow + Streamlit</div>
"""
render_html(dashboard)
