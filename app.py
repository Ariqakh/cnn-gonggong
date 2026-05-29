import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import base64
from io import BytesIO

st.set_page_config(
    page_title="Klasifikasi Jenis Gonggong",
    layout="centered"
)

# --- CSS STYLING UTAMA ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght=400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    margin: 0;
    padding: 0;
    height: 100%;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #A7FFFF 0%, #D1FFFF 100%) !important;
    min-height: 100vh !important;
}

[data-testid="stMain"] {
    background: transparent !important;
    padding-bottom: 20px !important; 
}

header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { 
    visibility: hidden; display: none !important; 
}

.navbar {
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 9999;
    background: #091a36;
    padding: 25px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.navbar-title {
    color: #ffffff;
    font-size: 20px;
    font-weight: 600;
}

.app-header {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 40px;
    margin-bottom: 12px;
    text-align: left;
}

.app-logo-img {
    width: 370px;
    height: 300px;
}
.app-title-container {
    display: flex;
    flex-direction: column;
}

.app-title-main {
    font-size: 65px; 
    font-weight: 800;
    color: #0b1d3a;
    line-height: 1.05;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.app-subtitle-main {
    font-size: 15px;
    color: #43647d;
    font-weight: 600;
    margin-top: 6px;
}

.img-preview-container {
    background: #E8E8E8;
    border-radius: 20px;
    margin: 35px auto;
    width: 100%;
    height: 300px; 
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    border: 1px solid #ddd;
}
.img-preview-container img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    border-radius: 10px;
}
.img-placeholder-text {
    font-size: 18px;
    color: #666;
    font-weight: 500;
}

/* ==========================================================================
   STYLE UNTUK UPLOADER DESKTOP & MOBILE (PENGUNCI HORIZONTAL)
   ========================================================================== */
[data-testid="stFileUploader"] section {
    background-color: #F3F3F3 !important;
    border: 1px solid #ccc !important;
    border-radius: 25px !important;
    padding: 12px !important;
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: flex-start !important;
}

/* Mematikan text bawaan asli Streamlit agar bersih */
[data-testid="stFileUploader"] section [data-testid="stUploadDropzone"] div {
    display: none !important;
}

/* Memaksa dropzone internal berjejer horizontal murni ke kanan */
[data-testid="stFileUploader"] [data-testid="stUploadDropzone"] {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: flex-start !important;
    gap: 12px !important;
    width: auto !important;
}

/* Menata tombol bawaan Streamlit (Browse files) */
[data-testid="stFileUploader"] section button[data-testid="stBaseButton-secondary"] {
    background-color: #FFFFFF !important;
    color: #333333 !important;
    border: 1px solid #CCCCCC !important;
    border-radius: 12px !important;
    padding: 8px 18px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    margin: 0 !important;
    display: inline-flex !important;
}

/* PROTEKSI DAN NORMALISASI BARIS SETELAH UPLOAD (KOLOM TIDAK LEBAR) */
[data-testid="stFileUploader"] [data-testid="stUploadDropzone"] + div {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    gap: 10px !important;
    margin-left: auto !important;
    padding: 0 !important;
}

/* Sembunyikan khusus tombol plus (+) bawaan */
[data-testid="stFileUploader"] button:has(svg path[d*="M19 "]) {
    display: none !important;
}

/* Kembalikan fungsionalitas dan tampilan tombol silang (X) */
[data-testid="stFileUploader"] button[aria-label="Remove file"] {
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}

/* BUTTON ANALISIS */
div.stButton > button {
    background-color: #2D6A6A !important;
    color: white !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    padding: 20px 40px !important;
    border-radius: 30px !important;
    border: none !important;
    margin: 10px auto !important;
    display: block !important;
    width: 200px;
}

.result-box {
    background-color: #87D4D4;
    border-radius: 20px;
    padding: 20px 25px;
    margin-top: 30px;
    text-align: left;
    display: flex;
    justify-content: flex-start;
    align-items: center;
}
.result-label {
    font-weight: 700;
    color: #0b1d3a;
    font-size: 18px;
    min-width: 180px; 
    display: inline-block;
}
.result-value {
    font-weight: 700;
    color: #0b1d3a;
    font-size: 18px;
}

.warning-box {
    background-color: #FFDADA;
    color: #CC0000;
    padding: 15px;
    border-radius: 15px;
    font-size: 14px;
    margin-top: 20px;
    margin-bottom: 10px;
    font-weight: 600;
    text-align: center;
    border: 1px solid #FFCCCC;
}

.page-wrapper {
    margin-bottom: 0px;
    padding-bottom: 0px;
}

.white-footer-canvas {
    position: relative !important;
    margin-top: 80px !important;
    padding: 20px 0px !important;
    display: flex !important;
    justify-content: center !important; 
    align-items: center !important;
    width: 100% !important;
}

.footer-text {
    text-align: center;
    color: #1a364a;
    font-size: 14px;
    font-weight: 500;
    line-height: 1.5;
    margin: 0 auto;
}

/* RESPONSIVE MOBILE OPTIMIZATION */
@media (max-width: 480px) {
    .navbar { padding: 15px 15px; gap: 8px; }
    .navbar-title { font-size: 14px; }
    .app-header { flex-direction: column; gap: 15px; text-align: center; margin-top: 40px; }
    .app-logo-img { width: 180px; height: auto; }
    .app-title-main { font-size: 32px; }
    .app-subtitle-main { font-size: 12px; }
    .img-preview-container { height: 220px; margin: 20px auto; }

    /* Penyelarasan kolom upload agar sama persis seperti versi website (tidak melebar) */
    [data-testid="stFileUploader"] section {
        display: inline-flex !important;  /* Mencegah kolom melar memenuhi layar HP */
        max-width: 100% !important;
        flex-direction: row !important; 
        align-items: center !important;
        justify-content: flex-start !important;
        border-radius: 25px !important; /* Samakan kebulatan border dengan desktop */
        padding: 10px !important;
    }
    
    [data-testid="stFileUploader"] [data-testid="stUploadDropzone"] {
        display: inline-flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: flex-start !important;
        flex-wrap: nowrap !important;
        gap: 10px !important;
        width: auto !important;
    }

    [data-testid="stFileUploader"] section button[data-testid="stBaseButton-secondary"] {
        padding: 6px 14px !important;
        font-size: 14px !important;
    }

    div.stButton > button {
        width: 100% !important;
        font-size: 18px !important;
        padding: 15px 20px !important;
    }
    
    .result-box {
        padding: 15px 20px;
        margin-top: 15px;
        flex-direction: row !important;
    }
    .result-label { font-size: 15px; min-width: 135px; }
    .result-value { font-size: 15px; }
    .white-footer-canvas { margin-top: 50px !important; padding: 15px 0px !important; }
    .footer-text { font-size: 11px; }
}
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL & DATA PREPARATION ---
@st.cache_resource
def load_my_model():
    from keras.models import load_model as keras_load_model
    from keras.layers import Dense, InputLayer, Dropout

    original_dense = Dense.from_config
    @classmethod
    def custom_dense(cls, config):
        config.pop("quantization_config", None)
        return original_dense(config)
    Dense.from_config = custom_dense

    original_input = InputLayer.from_config
    @classmethod
    def custom_input(cls, config):
        config.pop("batch_shape", None)
        config.pop("optional", None)
        if "batch_input_shape" not in config:
            config["batch_input_shape"] = [None, 224, 224, 3]
        return cls(**config)
    InputLayer.from_config = custom_input

    original_dropout = Dropout.from_config
    @classmethod
    def custom_dropout(cls, config):
        config.pop("seed_generator", None)
        return original_dropout(config)
    Dropout.from_config = custom_dropout

    model = keras_load_model("model_gonggong.h5", compile=False)
    return model

model = load_my_model()

classes = [
    "Canarium Mutabile",
    "Canarium Urseus",
    "Laevistrombus Turturella",
    "Pugilina Coclidium"
]

try:
    with open("logo_umrah.png", "rb") as f:
        nav_logo_bytes = f.read()
    encoded_nav_logo = base64.b64encode(nav_logo_bytes).decode()
except:
    encoded_nav_logo = ""

st.markdown(f"""
<div class="navbar">
    <div class="navbar-logo">
        <img src="data:image/png;base64,{encoded_nav_logo}" width="32" height="32" style="object-fit:cover;">
    </div>
    <span class="navbar-title">Universitas Maritim Raja Ali Haji</span>
</div>
<div class="navbar-spacer"></div>
""", unsafe_allow_html=True)

st.markdown("<div class='page-wrapper'>", unsafe_allow_html=True)

logo_html = ""
try:
    with open("logo_gonggong.png", "rb") as f:
        encoded_logo = base64.b64encode(f.read()).decode()
    logo_html = f"<img class='app-logo-img' src='data:image/png;base64,{encoded_logo}'>"
except:
    pass

st.markdown(f"""
<div class="app-header">
    {logo_html}
    <div class="app-title-container">
        <div class="app-title-main">Klasifikasi Jenis<br>Gonggong</div>
        <div class="app-subtitle-main">Berbasis Convolutional Neural Network (MobileNet)</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='main-card'>", unsafe_allow_html=True)

if "predicted_class" not in st.session_state:
    st.session_state.predicted_class = "-"
if "confidence_text" not in st.session_state:
    st.session_state.confidence_text = "-"
if "show_warning" not in st.session_state:
    st.session_state.show_warning = False

# --- RENDER FILE UPLOADER UTAMA ---
uploaded_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# METODE PSEUDO-ELEMENT JIKA FILE BELUM DIUPLOAD (TEKS DI SAMPING BUTTON)
if uploaded_file is None:
    st.markdown("""
    <style>
    [data-testid="stFileUploader"] [data-testid="stUploadDropzone"]::after {
        content: "200MB per file • JPG, PNG" !important;
        color: #555555 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
        white-space: nowrap !important;
        display: inline-block !important;
        margin-left: 5px !important;
    }
    @media (max-width: 480px) {
        [data-testid="stFileUploader"] [data-testid="stUploadDropzone"]::after {
            font-size: 13px !important;
            margin-left: 5px !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    st.markdown(f"""
    <div class='img-preview-container'>
        <img src="data:image/jpeg;base64,{img_str}">
    </div>
    """, unsafe_allow_html=True)
else:
    st.session_state.predicted_class = "-"
    st.session_state.confidence_text = "-"
    st.session_state.show_warning = False
    
    st.markdown("""
    <div class='img-preview-container'>
        <span class='img-placeholder-text'>Gambar</span>
    </div>
    """, unsafe_allow_html=True)

analyze_clicked = st.button("Analisis Gambar")

if analyze_clicked:
    if uploaded_file is not None:
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        max_conf = np.max(prediction)
        
        img_np = np.array(image)
        pure_white = np.sum(np.all(img_np >= 248, axis=-1))
        pure_black = np.sum(np.all(img_np <= 8, axis=-1))
        total_pixels = img_np.shape[0] * img_np.shape[1]
        extreme_ratio = (pure_white + pure_black) / total_pixels
        
        if extreme_ratio > 0.22 or max_conf < 0.50:
            st.session_state.show_warning = True
            st.session_state.predicted_class = ""
            st.session_state.confidence_text = ""
        else:
            st.session_state.show_warning = False
            predicted_index = np.argmax(prediction)
            st.session_state.predicted_class = classes[predicted_index]
            st.session_state.confidence_text = f"{max_conf * 100:.2f}%"
    else:
        st.warning("Silakan upload gambar terlebih dahulu.")

if st.session_state.show_warning:
    st.markdown("<div class='warning-box'>⚠️ Gambar tidak dikenali sebagai Gonggong. Harap upload foto Gonggong yang jelas.</div>", unsafe_allow_html=True)

st.markdown(f"""
<div class='result-box'>
    <span class='result-label'>Jenis Gonggong :</span>
    <span class='result-value'>{st.session_state.predicted_class}</span>
</div>
<div class='result-box'>
    <span class='result-label'>Tingkat Akurasi :</span>
    <span class='result-value'>{st.session_state.confidence_text}</span>
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True) 

st.markdown("""
<div class='white-footer-canvas'>
    <div class='footer-text'>
        © 2026 Sistem Klasifikasi Jenis Gonggong berbasis CNN MobileNet<br>
        Fakultas Teknik dan Teknologi Kemaritiman - UMRAH
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
