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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body {
    font-family: 'Inter', sans-serif;
    margin: 0;
    padding: 0;
    height: 100%;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #A7FFFF 0%, #D1FFFF 100%) !important;
    min-height: 100vh !important;
    display: flex !important;
    flex-direction: column !important;
}

[data-testid="stMain"] {
    background: transparent !important;
    padding-bottom: 0px !important; 
    display: flex !important;
    flex-direction: column !important;
    flex-grow: 1 !important;
}

[data-testid="stMainBlockContainer"] {
    display: flex !important;
    flex-direction: column !important;
    flex-grow: 1 !important;
    padding-bottom: 0px !important;
}

header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { 
    visibility: hidden; display: none !important; 
}

/* --- TAMBAHAN GAYA PETUNJUK --- */
.upload-instruction {
    text-align: center;
    color: #0b1d3a;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 10px;
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
    margin: 15px auto;
    width: 100%;
    height: 260px; 
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
    font-size: 16px;
    color: #666;
    font-weight: 500;
}

[data-testid="stFileUploader"] section {
    background-color: #F3F3F3 !important;
    border: 1px solid #ccc !important;
    border-radius: 30px !important;
    padding: 15px !important;
}

div[data-testid="stNotification"] {
    background-color: #f8d7da !important;
    border: 1px solid #f5c6cb !important;
    border-radius: 15px !important;
}
div[data-testid="stNotification"] p {
    color: #721c24 !important;
    font-weight: 700 !important;
    font-size: 14px !important;
}
div[data-testid="stNotification"] svg {
    fill: #721c24 !important;
}

.button-group {
    display: flex;
    flex-direction: row;
    justify-content: center;
    gap: 15px;
    margin: 20px auto;
    width: 100%;
}

div.stButton > button {
    color: white !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    padding: 15px 25px !important;
    border-radius: 30px !important;
    border: none !important;
    margin: 0 !important;
    display: block !important;
    width: 100%;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
}

div[data-testid="stButton"]:has(button[kind="secondary"]) > button,
div.stButton > button[kind="secondary"] {
    background: #0a3d3c !important;
}

div.stButton:nth-of-type(1) > button {
    background: #0a3d3c !important;
}

div.stButton > button#btn_remove_bg,
div.stButton > button[key="btn_remove_bg"] {
    background: #0a3d3c !important;
}

div.stButton > button#btn_analyze,
div.stButton > button[key="btn_analyze"] {
    background: #115c5a !important;
}

div.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(0,0,0,0.22) !important;
}

div.stButton > button:active {
    transform: translateY(0px) !important;
    opacity: 1 !important;
}

.result-box {
    background-color: #87D4D4;
    border-radius: 20px;
    padding: 20px 25px;
    margin-top: 20px;
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

.result-box-spacer {
    height: 100px;
    width: 100%;
}

.page-wrapper {
    display: flex !important;
    flex-direction: column !important;
    flex-grow: 1 !important;
    min-height: 100% !important;
    margin-bottom: 0px;
    padding-bottom: 0px;
}

.white-footer-canvas {
    position: relative !important;
    margin-top: auto !important; 
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

@media (max-width: 480px) {
    .navbar { padding: 15px 15px; gap: 8px; }
    .navbar-title { font-size: 14px; }
    .app-header { flex-direction: column; gap: 10px; text-align: center; margin-top: -10px; margin-bottom: 6px; }
    .app-logo-img { width: 195px; height: auto; }
    .app-title-main { font-size: 32px; }
    .app-subtitle-main { font-size: 12px; }
    .img-preview-container { height: 160px; margin: 8px auto; }
    .button-group { flex-direction: column !important; gap: 10px; margin: 10px auto !important; }
    
    [data-testid="stFileUploader"] section {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 15px !important;
        padding: 12px 20px !important;
        background-color: #EAEAEA !important;
        border: none !important;
        border-radius: 40px !important;
    }
    [data-testid="stFileUploader"] section svg { display: none !important; }
    [data-testid="stFileUploader"] section button {
        background-color: #FFFFFF !important;
        color: #333333 !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 12px !important;
        padding: 6px 16px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        margin: 0 !important;
        box-shadow: 0px 1px 3px rgba(0,0,0,0.1) !important;
    }
    [data-testid="stFileUploader"] section > input + div { display: none !important; }
    [data-testid="stFileUploader"] section::after {
        content: "200MB per file • JPG, PNG";
        font-size: 14px;
        color: #777777;
        font-weight: 400;
        display: inline-block;
    }
    div.stButton > button { width: 100% !important; font-size: 15px !important; padding: 12px 20px !important; }
    .result-box { padding: 15px 20px; margin-top: 12px; display: flex !important; flex-direction: row !important; }
    .result-label { font-size: 15px; min-width: 135px; margin-right: 0px; }
    .result-value { font-size: 15px; }
    .result-box-spacer { height: 100px; }
    .page-wrapper { margin-top: 15px; }
    .white-footer-canvas { margin-top: auto !important; padding: 15px 0px !important; }
    .footer-text { font-size: 11px; }
}
</style>
""", unsafe_allow_html=True)

# ... [load_my_model dan bagian lain tetap sama sampai File Uploader] ...

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

# ... [Setup Navbar] ...

st.markdown(f"""
<div class="navbar">
    <div class="navbar-logo">
        <img src="data:image/png;base64,{encoded_nav_logo}" width="32" height="32" style="object-fit:cover;">
    </div>
    <span class="navbar-title">Universitas Maritime Raja Ali Haji</span>
</div>
<div class="navbar-spacer"></div>
""", unsafe_allow_html=True)

st.markdown("<div class='page-wrapper'>", unsafe_allow_html=True)

# ... [Header Logo] ...

# --- FILE UPLOADER DENGAN PETUNJUK ---
st.markdown("<div class='upload-instruction'>📂 Klik atau Tarik (Drag) gambar Gonggong ke sini</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# ... [Sisa kode tetap sama] ...
