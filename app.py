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

# --- INJECT GLOBAL CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

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

/* DESKTOP UPLOADER */
[data-testid="stFileUploader"] section {
    background-color: #F3F3F3 !important;
    border: 1px solid #ccc !important;
    border-radius: 30px !important;
    padding: 15px !important;
}

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
    padding: 10px;
    border-radius: 15px;
    font-size: 13px;
    margin-bottom: 10px;
    font-weight: 600;
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

/* MOBILE RESPONSIVE OPTIMIZATION */
@media (max-width: 480px) {
    .navbar { padding: 15px 15px; gap: 8px; }
    .navbar-title { font-size: 14px; }
    .app-header { flex-direction: column; gap: 15px; text-align: center; margin-top: 40px; }
    .app-logo-img { width: 180px; height: auto; }
    .app-title-main { font-size: 32px; }
    .app-subtitle-main { font-size: 12px; }
    .img-preview-container { height: 220px; margin: 20px auto; }

    /* BOX UTAMA UPLOADER DI HP */
    [data-testid="stFileUploader"] section {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: space-between !important;
        padding: 12px 20px !important;
        background-color: #EAEAEA !important;
        border: none !important;
        border-radius: 40px !important;
    }
    
    /* Hilangkan ikon berkas bawaan upload */
    [data-testid="stFileUploader"] section svg { display: none !important; }
    
    /* Tombol Pilih File (Putih) */
    [data-testid="stFileUploader"] section button {
        background-color: #FFFFFF !important;
        color: #333333 !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 12px !important;
        padding: 6px 14px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        margin: 0 !important;
    }
    
    /* Sembunyikan teks seret bawaan browser */
    [data-testid="stFileUploader"] section > input + div { display: none !important; }
    
    /* Tampilkan teks instruksi ukuran file */
    [data-testid="stFileUploader"] section::after {
        content: "200MB per file • JPG, PNG";
        font-size: 13px;
        color: #777777;
        font-weight: 400;
        display: inline-block;
    }

    div.stButton > button { width: 100% !important; font-size: 18px !important; padding: 15px 20px !important; }
    .result-box { padding: 15px 20px; margin-top: 15px; display: flex !important; flex-direction: row !important; }
    .result-label { font-size: 15px; min-width: 135px; }
    .result-value { font-size: 15px; }
    .page-wrapper { margin-top: 25px; }
    .white-footer-canvas { margin-top: 50px !important; padding: 15px 0px !important; }
    .footer-text { font-size: 11px; }
}
</style>
""", unsafe_allow_html=True)

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

classes = ["Canarium Mutabile", "Canarium Urseus", "Laevistrombus Turturella", "Pugilina Coclidium"]

try:
    with open("logo_umrah.png", "rb") as f:
        encoded_nav_logo = base64.b64encode(f.read()).decode()
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
        logo_html = f"<img class='app-logo-img' src='data:image/png;base64,{base64.b64encode(f.read()).decode()}'>"
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

# --- STATE MANAGEMENT ---
if "pred_class" not in st.session_state:
    st.session_state.pred_class = "-"
if "conf_text" not in st.session_state:
    st.session_state.conf_text = "-"
if "warn_box_html" not in st.session_state:
    st.session_state.warn_box_html = ""
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# --- FILE UPLOADER WIDGET ---
uploaded_file = st.file_uploader(
    "Upload", 
    type=["jpg", "jpeg", "png"], 
    label_visibility="collapsed",
    key=f"file_uploader_{st.session_state.uploader_key}"
)

# ==========================================================================
# MANIPULASI ELEMENT KETIKA FILE BERHASIL DI-UPLOAD (MOBILE & DESKTOP)
# ==========================================================================
if uploaded_file is not None:
    # Sembunyikan panduan teks "200MB" & bersihkan sisa tanda tambah (+)
    st.markdown("""
    <style>
    [data-testid="stFileUploader"] section::after { content: "" !important; display: none !important; }
    [data-testid="stFileUploader"] section div { text-shadow: none !important; }
    
    @media (max-width: 480px) {
        /* Sembunyikan tombol 'Pilih File' bawaan saat terisi */
        [data-testid="stFileUploader"] section button { display: none !important; }
        
        /* Modifikasi tempat teks nama berkas agar rapi ke kiri */
        [data-testid="stFileUploader"] section > input + div {
            display: flex !important;
            flex-direction: row !important;
            align-items: center !important;
            justify-content: flex-start !important;
            width: 80% !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # MEMBUAT TOMBOL SILANG (X) RESET MANUAL YANG DIJAMIN MUNCUL DI HP
    # Menggunakan columns: kiri untuk nama file kustom, kanan untuk tombol reset silang merah (X)
    col_file, col_reset = st.columns([6, 1])
    with col_file:
        st.markdown(f"📁 <span style='font-size:14px; color:#333; font-weight:500;'>{uploaded_file.name}</span>", unsafe_allow_html=True)
    with col_reset:
        # Tombol X Merah tebal transparan bertindak sebagai pemicu reset total aplikasi
        if st.button("✕", help="Hapus gambar dan reset", key="clear_action_btn"):
            st.session_state.pred_class = "-"
            st.session_state.conf_text = "-"
            st.session_state.warn_box_html = ""
            st.session_state.uploader_key += 1  # Mengubah key otomatis membersihkan widget file_uploader
            st.rerun()

else:
    # Jika kosong, reset state otomatis ke awal
    st.session_state.pred_class = "-"
    st.session_state.conf_text = "-"
    st.session_state.warn_box_html = ""

# --- IMAGE PREVIEW CONTROLLER ---
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
    st.markdown("""
    <div class='img-preview-container'>
        <span class='img-placeholder-text'>Gambar</span>
    </div>
    """, unsafe_allow_html=True)

analyze_clicked = st.button("Analisis Gambar")

# --- PROSES MODEL & ANALISIS ---
if analyze_clicked:
    if uploaded_file is not None:
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        max_conf = np.max(prediction)
        
        img_np = np.array(image)
        pure_white = np.sum(np.all(img_np >= 245, axis=-1))
        total_pixels = img_np.shape[0] * img_np.shape[1]
        
        if max_conf < 0.65 or (pure_white / total_pixels) > 0.40:
            st.session_state.warn_box_html = "<div class='warning-box'>⚠️ Gambar tidak dikenali sebagai Gonggong. Harap upload foto Gonggong yang jelas.</div>"
            st.session_state.pred_class = "-"
            st.session_state.conf_text = "-"
        else:
            st.session_state.warn_box_html = ""
            predicted_index = np.argmax(prediction)
            st.session_state.pred_class = classes[predicted_index]
            st.session_state.conf_text = f"{max_conf * 100:.2f}%"
    else:
        st.warning("Silakan upload gambar terlebih dahulu.")

if st.session_state.warn_box_html:
    st.markdown(st.session_state.warn_box_html, unsafe_allow_html=True)

# --- SHOW PREDICTION RESULT ---
st.markdown(f"""
<div class='result-box'>
    <span class='result-label'>Jenis Gonggong :</span>
    <span class='result-value'>{st.session_state.pred_class}</span>
</div>
<div class='result-box'>
    <span class='result-label'>Tingkat Akurasi :</span>
    <span class='result-value'>{st.session_state.conf_text}</span>
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
