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
html { scroll-behavior: smooth; }
body { font-family: 'Inter', sans-serif; margin: 0; padding: 0; height: 100%; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #A7FFFF 0%, #D1FFFF 100%) !important;
    min-height: 100vh !important;
    display: flex !important;
    flex-direction: column !important;
}

[data-testid="stMain"] { background: transparent !important; flex-grow: 1 !important; }
[data-testid="stMainBlockContainer"] { display: flex !important; flex-direction: column !important; flex-grow: 1 !important; }

header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { 
    visibility: hidden; display: none !important; 
}

.navbar {
    position: fixed; top: 0; left: 0; right: 0; z-index: 9999; background: #091a36;
    padding: 25px 20px; display: flex; align-items: center; gap: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}
.navbar-title { color: #ffffff; font-size: 20px; font-weight: 600; }

.welcome-container {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    text-align: center; min-height: 80vh; padding: 20px; animation: fadeIn 1s ease-out;
}

.welcome-title { font-size: 50px; font-weight: 800; color: #0b1d3a; line-height: 1.1; margin-bottom: 20px; text-transform: uppercase; }
.welcome-subtitle { font-size: 18px; color: #43647d; font-weight: 500; max-width: 600px; margin-bottom: 40px; }

.cta-scroll-button {
    background: #0a3d3c; color: white !important; padding: 16px 36px; border-radius: 35px;
    font-size: 16px; font-weight: 700; text-decoration: none !important;
    box-shadow: 0 10px 25px rgba(10, 61, 60, 0.3); transition: all 0.3s;
}

.app-header { display: flex; align-items: center; justify-content: center; gap: 40px; margin: 40px 0; }
.app-logo-img { width: 300px; }
.app-title-main { font-size: 50px; font-weight: 800; color: #0b1d3a; line-height: 1; text-transform: uppercase; }

.img-preview-container {
    background: #E8E8E8; border-radius: 20px; margin: 15px auto; width: 100%; height: 260px;
    display: flex; align-items: center; justify-content: center; overflow: hidden; border: 1px solid #ddd;
}

.result-box { background-color: #87D4D4; border-radius: 20px; padding: 20px; margin-top: 20px; }

@media (max-width: 480px) {
    .welcome-title { font-size: 32px; }
    .app-header { flex-direction: column; text-align: center; }
    .app-logo-img { width: 150px; }
    .app-title-main { font-size: 28px; }
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_my_model():
    from keras.models import load_model
    return load_model("model_gonggong.h5", compile=False)

model = load_my_model()
classes = ['Canarium Mutabile', 'Canarium Urseus', 'Laevistrombus Turturella', 'Pugilina Coclidium']

# Inisialisasi state
if "show_app" not in st.session_state: st.session_state.show_app = False

# Navbar
st.markdown("""
<div class="navbar"><span class="navbar-title">Universitas Maritim Raja Ali Haji</span></div>
""", unsafe_allow_html=True)

# Jika belum di-klik, tampilkan Welcome
if not st.session_state.show_app:
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-title">Selamat Datang di<br>Sistem Klasifikasi Gonggong</div>
        <div class="welcome-subtitle">Identifikasi jenis siput Gonggong khas Kepulauan Riau menggunakan AI.</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Klasifikasi Sekarang", key="btn_start"):
        st.session_state.show_app = True
        st.rerun()

# Jika sudah di-klik, tampilkan isi aplikasi
else:
    st.markdown("<div id='main-app'>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="app-header">
        <div class="app-title-container">
            <div class="app-title-main">Klasifikasi Jenis<br>Gonggong</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    @st.fragment
    def application_core():
        uploaded_file = st.file_uploader("Upload Foto", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar Anda", use_container_width=True)
            if st.button("Analisis"):
                st.write("Hasil Klasifikasi...") # Logika model di sini

    application_core()
    
    if st.button("Kembali ke Beranda"):
        st.session_state.show_app = False
        st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)
