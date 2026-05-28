import streamlit as st
from PIL import Image
import numpy as np

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Klasifikasi Jenis Gonggong - UMRAH",
    layout="centered"
)

# =========================
# CUSTOM CSS — PERSIS SESUAI SCREENSHOT
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Righteous&display=swap');

/* ===================== RESET & BASE ===================== */
* { box-sizing: border-box; margin: 0; padding: 0; }

/* Background cyan gradient */
.stApp {
    background: linear-gradient(160deg, #a8f0ee 0%, #7de8e4 40%, #b2f5f2 100%);
    min-height: 100vh;
    font-family: 'Nunito', sans-serif;
}

/* Sembunyikan header bawaan Streamlit */
header { visibility: hidden; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Hapus padding default block container */
.block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    max-width: 100% !important;
}

/* ===================== TOP BAR ===================== */
.top-bar {
    background: #1a2a4a;
    color: #ffffff;
    padding: 10px 24px;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 9999;
    display: flex;
    align-items: center;
    gap: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
}

.top-bar img {
    height: 38px;
    width: auto;
}

.top-bar-text {
    font-family: 'Nunito', sans-serif;
    font-size: 17px;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: #ffffff;
}

/* ===================== KONTEN UTAMA ===================== */
.main-wrapper {
    margin-top: 60px;
    padding: 30px 16px 0 16px;
    display: flex;
    flex-direction: column;
    align-items: center;
}

/* ===================== HEADER LOGO + JUDUL ===================== */
.brand-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 18px;
    margin-bottom: 10px;
}

.brand-row img {
    height: 110px;
    width: auto;
    filter: drop-shadow(0px 4px 12px rgba(0,0,0,0.15));
}

.brand-title {
    font-family: 'Righteous', cursive;
    font-size: 40px;
    font-weight: 400;
    color: #0d2b52;
    line-height: 1.15;
    text-transform: uppercase;
    letter-spacing: 1px;
    text-shadow: 1px 2px 0px rgba(255,255,255,0.3);
}

.brand-subtitle {
    text-align: center;
    color: #1a5a72;
    font-size: 13.5px;
    font-weight: 600;
    letter-spacing: 1.5px;
    margin-bottom: 28px;
}

/* ===================== CARD UTAMA ===================== */
.main-card {
    background: rgba(255,255,255,0.55);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border-radius: 24px;
    border: 1.5px solid rgba(255,255,255,0.8);
    box-shadow: 0 8px 32px rgba(0, 100, 120, 0.08);
    padding: 30px 32px 30px 32px;
    width: 100%;
    max-width: 480px;
    margin-bottom: 10px;
}

/* ===================== FILE UPLOADER ===================== */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.7);
    border: 1.5px dashed #7fbfbe;
    border-radius: 12px;
    padding: 6px 12px;
}

[data-testid="stFileUploader"] label { display: none !important; }

/* ===================== PLACEHOLDER GAMBAR ===================== */
.img-placeholder {
    background: #f0f0f0;
    border: 1.5px solid #d8d8d8;
    border-radius: 12px;
    width: 100%;
    height: 230px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #888;
    font-size: 15px;
    font-family: 'Nunito', sans-serif;
    margin: 20px 0;
}

/* ===================== PREVIEW GAMBAR ===================== */
.image-preview-wrapper {
    display: flex;
    justify-content: center;
    margin: 20px 0;
}

.image-preview-wrapper img {
    border-radius: 12px;
    border: 2px solid #ffffff;
    box-shadow: 0 6px 20px rgba(0,80,100,0.12);
    max-width: 280px;
}

/* ===================== TOMBOL ANALISIS ===================== */
div.stButton > button:first-child {
    background: #5b9fa0;
    color: #ffffff;
    font-family: 'Nunito', sans-serif;
    font-size: 15px;
    font-weight: 800;
    letter-spacing: 1px;
    padding: 13px 40px;
    border-radius: 50px;
    border: none;
    display: block;
    margin: 0 auto;
    box-shadow: 0 4px 18px rgba(50, 130, 140, 0.35);
    transition: all 0.2s ease;
    min-width: 200px;
}

div.stButton > button:first-child:hover {
    background: #4a8c8d;
    transform: translateY(-2px);
    box-shadow: 0 6px 22px rgba(50, 130, 140, 0.45);
}

/* ===================== PANEL HASIL ===================== */
.result-section {
    margin-top: 24px;
}

.result-label-row {
    background: #5b9fa0;
    border-radius: 50px;
    padding: 12px 22px;
    color: #ffffff;
    font-family: 'Nunito', sans-serif;
    font-size: 15px;
    font-weight: 800;
    letter-spacing: 0.5px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ===================== FOOTER ===================== */
.app-footer {
    text-align: center;
    color: #3a5a6a;
    font-size: 12.5px;
    font-family: 'Nunito', sans-serif;
    line-height: 1.9;
    padding: 30px 0 24px 0;
    width: 100%;
}

/* Sembunyikan label file uploader */
[data-testid="stFileUploader"] section {
    padding: 0 !important;
}

/* Pastikan gambar streamlit responsive di dalam card */
[data-testid="stImage"] {
    display: flex;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TOP BAR
# =========================
try:
    import base64, pathlib
    logo_umrah_path = pathlib.Path("logo_umrah.png")
    if logo_umrah_path.exists():
        logo_umrah_b64 = base64.b64encode(logo_umrah_path.read_bytes()).decode()
        logo_umrah_src = f"data:image/png;base64,{logo_umrah_b64}"
    else:
        logo_umrah_src = "logo_umrah.png"
except Exception:
    logo_umrah_src = "logo_umrah.png"

st.markdown(f"""
<div class="top-bar">
    <img src="{logo_umrah_src}" onerror="this.style.display='none'">
    <div class="top-bar-text">Universitas Maritim Raja Ali Haji</div>
</div>
""", unsafe_allow_html=True)

# =========================
# BUKA WRAPPER UTAMA
# =========================
st.markdown("<div class='main-wrapper'>", unsafe_allow_html=True)

# =========================
# LOGO GONGGONG + JUDUL
# =========================
try:
    import base64, pathlib
    logo_gonggong_path = pathlib.Path("logo_gonggong.png")
    if logo_gonggong_path.exists():
        logo_gonggong_b64 = base64.b64encode(logo_gonggong_path.read_bytes()).decode()
        logo_gonggong_src = f"data:image/png;base64,{logo_gonggong_b64}"
    else:
        logo_gonggong_src = "logo_gonggong.png"
except Exception:
    logo_gonggong_src = "logo_gonggong.png"

st.markdown(f"""
<div class="brand-row">
    <img src="{logo_gonggong_src}" onerror="this.style.display='none'">
    <div class="brand-title">Klasifikasi Jenis<br>Gonggong</div>
</div>
<div class="brand-subtitle">Berbasis Convolutional Neural Network (MobileNet)</div>
""", unsafe_allow_html=True)

# =========================
# CARD UTAMA — BUKA
# =========================
st.markdown("<div class='main-card'>", unsafe_allow_html=True)

# =========================
# FILE UPLOADER
# =========================
uploaded_file = st.file_uploader(
    "Upload gambar gonggong",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# =========================
# PREVIEW GAMBAR / PLACEHOLDER
# =========================
if uploaded_file is None:
    st.markdown("""
    <div class="img-placeholder">Gambar</div>
    """, unsafe_allow_html=True)
else:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='image-preview-wrapper'>", unsafe_allow_html=True)
        st.image(image, width=280)
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# TOMBOL ANALISIS
# =========================
col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
with col_b2:
    action_trigger = st.button("Analisis Gambar")

# =========================
# PANEL HASIL (Selalu tampil, kosong dulu)
# =========================
predicted_class_text = ""
confidence_text = ""

# Eksekusi Model
if action_trigger and uploaded_file is not None:
    try:
        from keras.models import load_model as keras_load
        from keras.layers import Dense

        @st.cache_resource
        def load_gonggong_model():
            original_from_config = Dense.from_config

            @classmethod
            def custom_from_config(cls, config):
                config.pop("quantization_config", None)
                return original_from_config(config)

            Dense.from_config = custom_from_config
            return keras_load("model_gonggong.h5", compile=False)

        model = load_gonggong_model()

        classes = [
            "Canarium Mutabile",
            "Canarium Urseus",
            "Laevistrombus Turturella",
            "Pugilina Coclidium"
        ]

        img = image.resize((224, 224))
        img_array = np.array(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class_text = classes[predicted_index]
        confidence_text = f"{np.max(prediction) * 100:.2f}%"

    except Exception as e:
        predicted_class_text = f"Error: {e}"
        confidence_text = "-"

elif action_trigger and uploaded_file is None:
    st.warning("Silakan unggah gambar terlebih dahulu.")

# Render kotak hasil
st.markdown(f"""
<div class="result-section">
    <div class="result-label-row">Jenis Gonggong : {predicted_class_text}</div>
    <div class="result-label-row">Tingkat Akurasi : {confidence_text}</div>
</div>
""", unsafe_allow_html=True)

# =========================
# CARD UTAMA — TUTUP
# =========================
st.markdown("</div>", unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="app-footer">
    © 2026 Sistem Klasifikasi Jenis Gonggong berbasis CNN MobileNet<br>
    Fakultas Teknik dan Teknologi Kemaritiman – UMRAH
</div>
""", unsafe_allow_html=True)

# Tutup main wrapper
st.markdown("</div>", unsafe_allow_html=True)