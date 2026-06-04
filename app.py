import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import base64
from io import BytesIO
from rembg import remove

st.set_page_config(
    page_title="Klasifikasi Jenis Gonggong",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #A7FFFF 0%, #D1FFFF 100%) !important;
    min-height: 100vh !important;
}

header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { 
    visibility: hidden; display: none !important; 
}

/* Navbar & Typography */
.navbar { position: fixed; top: 0; left: 0; right: 0; z-index: 9999; background: #091a36; padding: 25px 20px; display: flex; align-items: center; gap: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
.navbar-title { color: #ffffff; font-size: 20px; font-weight: 600; }

.welcome-container { display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; min-height: 100vh; padding: 20px; }
.welcome-title { font-size: 56px; font-weight: 800; color: #0b1d3a; line-height: 1.1; margin-bottom: 15px; text-transform: uppercase; }
.welcome-subtitle { font-size: 18px; color: #43647d; font-weight: 500; max-width: 600px; margin-bottom: 40px; }

.cta-scroll-button { background: #0a3d3c; color: white !important; padding: 16px 36px; border-radius: 35px; font-size: 16px; font-weight: 700; text-decoration: none !important; box-shadow: 0 10px 25px rgba(10, 61, 60, 0.3); }

/* Main Content Area */
.main-classification-section { padding-top: 80px; }

.app-header { display: flex; align-items: center; justify-content: center; gap: 40px; margin: 40px 0; }
.app-logo-img { width: 370px; height: auto; }
.app-title-main { font-size: 65px; font-weight: 800; color: #0b1d3a; text-transform: uppercase; }

.img-preview-container { background: #E8E8E8; border-radius: 20px; margin: 15px auto; width: 100%; height: 260px; display: flex; align-items: center; justify-content: center; border: 1px solid #ddd; }
.img-preview-container img { max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 10px; }

/* Mobile Specific Tweaks */
@media (max-width: 480px) {
    .welcome-title { font-size: 32px; }
    .welcome-subtitle { font-size: 14px; }
    .app-header { flex-direction: column; gap: 10px; }
    .app-logo-img { width: 200px; }
    .app-title-main { font-size: 28px; }
    .img-preview-container { height: 200px; }
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_my_model():
    from keras.models import load_model
    return load_model("model_gonggong.h5", compile=False)

model = load_my_model()
classes = ['Canarium Mutabile', 'Canarium Urseus', 'Laevistrombus Turturella', 'Pugilina Coclidium']

# Navbar
st.markdown("""
<div class="navbar">
    <span class="navbar-title">Universitas Maritim Raja Ali Haji</span>
</div>
""", unsafe_allow_html=True)

# Welcome Section (Selalu tampil di awal)
st.markdown("""
<div class="welcome-container">
    <div class="welcome-title">Selamat Datang di<br>Sistem Klasifikasi Gonggong</div>
    <div class="welcome-subtitle">Identifikasi jenis siput Gonggong khas Kepulauan Riau menggunakan AI.</div>
    <a class="cta-scroll-button" href="#mulai-klasifikasi">Mulai Klasifikasi</a>
</div>
""", unsafe_allow_html=True)

# Classification Section
st.markdown("<div id='mulai-klasifikasi' class='main-classification-section'>", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <div class="app-title-container">
        <div class="app-title-main">Klasifikasi Jenis Gonggong</div>
    </div>
</div>
""", unsafe_allow_html=True)

@st.fragment
def application_core():
    if "bg_removed_image" not in st.session_state: st.session_state.bg_removed_image = None
    if "pred_class" not in st.session_state: st.session_state.pred_class = "-"
    if "conf_text" not in st.session_state: st.session_state.conf_text = "-"

    uploaded_file = st.file_uploader("Upload Foto", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        # Logika pemrosesan sama seperti sebelumnya...
        if st.button("Hapus Latar Belakang"):
            output = remove(image)
            st.session_state.bg_removed_image = output
            st.rerun()
            
        if st.session_state.bg_removed_image:
            st.image(st.session_state.bg_removed_image)
            if st.button("Analisis"):
                # (Tambahkan logika prediksi Anda di sini)
                st.write(f"Hasil: {st.session_state.pred_class}")

application_core()

st.markdown("</div>", unsafe_allow_html=True) # Tutup main-classification-section
