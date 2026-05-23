import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import os
import gdown

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_PATH  = "fruit_model_001.keras"
MODEL_URL   = "https://drive.google.com/file/d/10qcD2PvWf0qt3k9Cy39kJJgjVQhFZgLy/view?usp=sharing"
IMG_SIZE    = 128
NUM_CLASSES = 36

CLASS_NAMES = [
    "apple", "banana", "beetroot", "bell pepper", "cabbage", "capsicum",
    "carrot", "cauliflower", "chilli pepper", "corn", "cucumber", "eggplant",
    "garlic", "ginger", "grapes", "jalepeno", "kiwi", "lemon", "lettuce",
    "mango", "onion", "orange", "paprika", "pear", "peas", "pineapple",
    "pomegranate", "potato", "raddish", "soy beans", "spinach", "sweetcorn",
    "sweetpotato", "tomato", "turnip", "watermelon",
]

CLASS_EMOJI = {
    "apple": "🍎", "banana": "🍌", "carrot": "🥕", "corn": "🌽",
    "cucumber": "🥒", "eggplant": "🍆", "garlic": "🧄", "grapes": "🍇",
    "kiwi": "🥝", "lemon": "🍋", "lettuce": "🥬", "mango": "🥭",
    "onion": "🧅", "orange": "🍊", "paprika": "🫑", "pear": "🍐",
    "pineapple": "🍍", "pomegranate": "🫐", "potato": "🥔", "spinach": "🥬",
    "tomato": "🍅", "watermelon": "🍉", "peas": "🫛", "beetroot": "🫚",
}

# ── Model download ──────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    with st.spinner("📦 Downloading model weights…"):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fresh Detect · Fruit & Veg Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ── App background ── */
    .stApp {
        background: #F7FAF4;
    }

    /* ── Header strip ── */
    .app-header {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 1.1rem 0 0.8rem;
        border-bottom: 1px solid #D0E6BF;
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        font-weight: 600;
        color: #1D3D0A;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .app-header .tagline {
        font-size: 0.82rem;
        color: #5A7A42;
        margin: 0;
        letter-spacing: 0.04em;
    }
    .model-ready-pill {
        margin-left: auto;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #EAF3DE;
        border: 1px solid #97C459;
        border-radius: 99px;
        padding: 4px 14px;
        font-size: 0.75rem;
        color: #3B6D11;
        font-weight: 500;
    }

    /* ── Prediction hero card ── */
    .pred-hero {
        background: #ffffff;
        border: 1px solid #D0E6BF;
        border-radius: 16px;
        padding: 1.4rem 1.6rem 1.2rem;
        margin-bottom: 14px;
    }
    .pred-hero .label {
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #7FAB5B;
        margin-bottom: 4px;
    }
    .pred-hero .class-name {
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        font-weight: 600;
        color: #1D3D0A;
        text-transform: capitalize;
        line-height: 1.1;
        margin-bottom: 14px;
    }
    .conf-row {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .conf-bar-bg {
        flex: 1;
        height: 9px;
        background: #EAF3DE;
        border-radius: 99px;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 99px;
    }
    .conf-pct {
        font-size: 1.1rem;
        font-weight: 500;
        min-width: 56px;
        text-align: right;
        color: #1D3D0A;
    }

    /* ── Top-K rows ── */
    .topk-card {
        background: #ffffff;
        border: 1px solid #D0E6BF;
        border-radius: 16px;
        padding: 1.1rem 1.4rem;
    }
    .topk-card h3 {
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #7FAB5B;
        margin-bottom: 12px;
    }
    .topk-row {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 5px 0;
        border-bottom: 0.5px solid #EAF3DE;
    }
    .topk-row:last-child { border-bottom: none; }
    .rank-badge {
        font-size: 0.7rem;
        color: #9EB57E;
        min-width: 16px;
        text-align: right;
    }
    .topk-name {
        flex: 1;
        font-size: 0.84rem;
        color: #2A4A18;
        text-transform: capitalize;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .topk-name.winner { font-weight: 500; color: #3B6D11; }
    .topk-mini-bar {
        width: 80px;
        height: 5px;
        background: #EAF3DE;
        border-radius: 99px;
        overflow: hidden;
    }
    .topk-mini-fill { height: 100%; border-radius: 99px; }
    .topk-pct {
        font-size: 0.78rem;
        color: #7FAB5B;
        min-width: 38px;
        text-align: right;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #F0F7E8;
        border-right: 1px solid #D0E6BF;
    }
    .sidebar-info-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.82rem;
        padding: 5px 0;
        border-bottom: 0.5px solid #D0E6BF;
        color: #2A4A18;
    }
    .sidebar-info-row span:first-child { color: #7FAB5B; }

    /* ── Uploader tweaks ── */
    [data-testid="stFileUploader"] {
        border-radius: 14px;
        border: 2px dashed #97C459 !important;
        background: #F7FAF4;
    }
    [data-testid="stFileUploader"]:hover {
        background: #EAF3DE;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #D0E6BF;
        border-radius: 12px;
        padding: 0.7rem 1rem;
    }
    [data-testid="stMetricLabel"] { color: #7FAB5B !important; }
    [data-testid="stMetricValue"] { color: #1D3D0A !important; font-family: 'Playfair Display', serif !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-header">
      <div>
        <h1>🌿 Fresh Detect</h1>
        <p class="tagline">MobileNetV2 · 36 classes · 128 × 128 px</p>
      </div>
      <span class="model-ready-pill">● Model ready</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Settings")
    top_k = st.slider("Top-K predictions", min_value=3, max_value=10, value=5, step=1)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ℹ️ Model info")

    info = {
        "Architecture": "MobileNetV2",
        "Input size": f"{IMG_SIZE} × {IMG_SIZE}",
        "Classes": str(NUM_CLASSES),
        "Format": ".keras",
    }
    for k, v in info.items():
        st.markdown(
            f'<div class="sidebar-info-row"><span>{k}</span><span>{v}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🌱 All classes")
    cols = st.columns(2)
    for i, name in enumerate(CLASS_NAMES):
        em = CLASS_EMOJI.get(name, "·")
        cols[i % 2].markdown(
            f"<small style='color:#5A7A42'>{em} {name}</small>", unsafe_allow_html=True
        )

# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(model, image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.expand_dims(np.array(img, dtype=np.float32), 0)
    return model.predict(arr, verbose=0)[0]


def conf_color(pct: float) -> str:
    if pct >= 70:
        return "#639922"
    if pct >= 40:
        return "#BA7517"
    return "#E24B4A"

# ── Upload area ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "📷 Drop an image here or click to browse",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="visible",
)

if uploaded:
    image = Image.open(uploaded)

    with st.spinner("Analysing image…"):
        preds = predict(model, image)

    top_idx   = int(np.argmax(preds))
    top_class = CLASS_NAMES[top_idx]
    top_conf  = float(preds[top_idx]) * 100
    top_em    = CLASS_EMOJI.get(top_class, "🌱")
    bar_color = conf_color(top_conf)

    # ── Two-column layout ─────────────────────────────────────────────────────
    col_img, col_res = st.columns([1, 1.35], gap="large")

    with col_img:
        st.image(image, use_container_width=True)

    with col_res:
        # ── Prediction hero ───────────────────────────────────────────────────
        st.markdown(
            f"""
            <div class="pred-hero">
              <div class="label">Predicted class</div>
              <div class="class-name">{top_em} {top_class}</div>
              <div class="conf-row">
                <div class="conf-bar-bg">
                  <div class="conf-bar-fill" style="width:{top_conf:.1f}%;background:{bar_color}"></div>
                </div>
                <div class="conf-pct">{top_conf:.1f}%</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Top-K list ────────────────────────────────────────────────────────
        top_k_idx   = np.argsort(preds)[::-1][:top_k]
        top_k_names = [CLASS_NAMES[i] for i in top_k_idx]
        top_k_confs = [float(preds[i]) * 100 for i in top_k_idx]
        max_conf    = top_k_confs[0]

        rows_html = ""
        for rank, (name, conf) in enumerate(zip(top_k_names, top_k_confs), 1):
            is_top   = rank == 1
            bar_w    = round(conf / max_conf * 100)
            fill_col = bar_color if is_top else "#C5D9AA"
            cls      = "topk-name winner" if is_top else "topk-name"
            em       = CLASS_EMOJI.get(name, "")
            rows_html += f"""
              <div class="topk-row">
                <span class="rank-badge">{rank}</span>
                <span class="{cls}">{em} {name}</span>
                <div class="topk-mini-bar">
                  <div class="topk-mini-fill" style="width:{bar_w}%;background:{fill_col}"></div>
                </div>
                <span class="topk-pct">{conf:.1f}%</span>
              </div>"""

        st.markdown(
            f"""
            <div class="topk-card">
              <h3>Top {top_k} predictions</h3>
              {rows_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Plotly bar chart ──────────────────────────────────────────────────────
    with st.expander(f"📊 Full confidence chart — top {top_k}"):
        fig = go.Figure(
            go.Bar(
                x=top_k_confs[::-1],
                y=[f"{CLASS_EMOJI.get(n,'')} {n}" for n in top_k_names[::-1]],
                orientation="h",
                marker=dict(
                    color=[
                        bar_color if n == top_class else "#97C459"
                        for n in top_k_names[::-1]
                    ],
                    line=dict(color="rgba(0,0,0,0)", width=0),
                ),
                text=[f"{c:.1f}%" for c in top_k_confs[::-1]],
                textposition="outside",
                textfont=dict(size=12, color="#3B6D11"),
            )
        )
        fig.update_layout(
            margin=dict(l=0, r=60, t=10, b=10),
            xaxis=dict(
                title="Confidence (%)",
                range=[0, min(max_conf * 1.25, 105)],
                tickfont=dict(size=11, color="#7FAB5B"),
                gridcolor="#EAF3DE",
                zeroline=False,
            ),
            yaxis=dict(title="", tickfont=dict(size=12, color="#2A4A18")),
            height=max(240, top_k * 44),
            plot_bgcolor="#F7FAF4",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", size=12),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── All class scores ──────────────────────────────────────────────────────
    with st.expander("🔍 All class scores"):
        all_idx = np.argsort(preds)[::-1]
        cols = st.columns(3)
        for j, i in enumerate(all_idx):
            em = CLASS_EMOJI.get(CLASS_NAMES[i], "·")
            cols[j % 3].markdown(
                f"<small style='color:#5A7A42'>{em} **{CLASS_NAMES[i]}** — `{preds[i]*100:.2f}%`</small>",
                unsafe_allow_html=True,
            )

else:
    st.markdown(
        """
        <div style="text-align:center;padding:3rem 1rem;color:#7FAB5B;font-size:0.9rem;">
          Upload an image above to see predictions ↑
        </div>
        """,
        unsafe_allow_html=True,
    )