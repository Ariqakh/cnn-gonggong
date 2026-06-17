import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import base64
from io import BytesIO
from rembg import remove
import os

# ==============================================================================
# 1. KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="Klasifikasi Jenis Gonggong - UMRAH",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# 2. CSS CUSTOM (STYLING)
# ==============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html { scroll-behavior: smooth; }

body {
    font-family: 'Inter', sans-serif;
    margin: 0;
    padding: 0;
    height: 100%;
}

/* Background gradient container */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #A7FFFF 0%, #D1FFFF 100%) !important;
    min-height: 100vh !important;
    display: flex !important;
    flex-direction: column !important;
}

/* Main container overrides */
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

/* Hide Streamlit default branding */
header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { 
    visibility: hidden; display: none !important; 
}

/* Navbar Component */
.navbar {
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 9999;
    background: #091a36;
    padding: 20px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}
.navbar-title {
    color: #ffffff;
    font-size: 18px;
    font-weight: 600;
}

/* Welcome Section */
.welcome-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    min-height: 70vh;
    padding: 80px 20px 20px 20px;
    animation: fadeIn 1s ease-out;
}

.welcome-title {
    font-size: 48px;
    font-weight: 800;
    color: #0b1d3a;
    line-height: 1.1;
    margin-bottom: 15px;
    text-transform: uppercase;
}

.welcome-subtitle {
    font-size: 16px;
    color: #43647d;
    font-weight: 500;
    max-width: 600px;
    margin-bottom: 40px;
    line-height: 1.6;
}

.cta-scroll-button {
    background: #0a3d3c;
    color: white !important;
    padding: 16px 36px;
    border-radius: 35px;
    font-size: 16px;
    font-weight: 700;
    text-decoration: none !important;
    box-shadow: 0 10px 25px rgba(10, 61, 60, 0.3);
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    display: inline-block;
}
.cta-scroll-button:hover {
    transform: translateY(-4px);
    box-shadow: 0 15px 30px rgba(10, 61, 60, 0.4);
    background: #115c5a;
}

/* App Header Section */
.app-header {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 40px;
    margin: 60px 0 12px 0;
    text-align: left;
    padding-top: 40px;
}
.app-logo-img { width: 300px; height: auto; }
.app-title-main { font-size: 50px; font-weight: 800; color: #0b1d3a; line-height: 1.05; text-transform: uppercase; }
.app-subtitle-main { font-size: 14px; color: #43647d; font-weight: 600; margin-top: 6px; }

/* Image Preview Container */
.img-preview-container {
    background: #E8E8E8;
    border-radius: 20px;
    margin: 15px auto;
    width: 100%;
    height: 250px; 
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

/* File Uploader Style */
[data-testid="stFileUploader"] section {
    background-color: #F3F3F3 !important;
    border: 1px solid #ccc !important;
    border-radius: 30px !important;
    padding: 15px !important;
}

/* Button Group Style */
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
    width: 100%;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
}

/* Button Colors */
div.stButton > button[kind="secondary"],
div.stButton:nth-of-type(1) > button { background: #0a3d3c !important; }
div.stButton > button[key*="anlz"] { background: #115c5a !important; }
div.stButton > button[key*="reset"] { background: #64748b !important; }

/* Result Box */
.result-box {
    background-color: #87D4D4;
    border-radius: 20px;
    padding: 20px 25px;
    margin-top: 20px;
    display: flex;
    justify-content: flex-start;
    align-items: center;
}
.result-label { font-weight: 700; color: #0b1d3a; font-size: 18px; min-width: 180px; }
.result-value { font-weight: 700; color: #0b1d3a; font-size: 18px; }

/* Footer */
.white-footer-canvas { padding: 40px 0px !important; text-align: center; color: #1a364a; }

/* Mobile Responsiveness */
@media (max-width: 480px) {
    .app-header { flex-direction: column; }
    .welcome-title { font-size: 32px; }
}

@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. FUNGSI PEMUATAN MODEL
# ==============================================================================
@st.cache_resource
def load_my_model():
    """
    Fungsi untuk memuat model MobileNet yang telah dilatih.
    Karena model ini menggunakan layer custom (Lambda), kita memuat model
    secara langsung. Pastikan 'model_gonggong.h5' tersedia di direktori.
    """
    try:
        # Karena kita menggunakan arsitektur MobileNet dengan Lambda,
        # kita memuat model tanpa mengompilasi ulang untuk efisiensi di streamlit
        model = tf.keras.models.load_model("model_gonggong.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None

# Inisialisasi Model
model = load_my_model()

# Daftar Kelas (Sesuaikan dengan urutan label training Anda)
classes = [
    'Canarium Mutabile', 
    'Canarium Urseus', 
    'Laevistrombus Turturella', 
    'Pugilina Coclidium'
]

# ==============================================================================
# 4. LOGIKA APLIKASI (CORE)
# ==============================================================================
@st.fragment
def application_core():
    """
    Fragment aplikasi yang memproses gambar dan melakukan inferensi.
    Fragment digunakan agar UI tidak mereload seluruh halaman saat interaksi.
    """
    # Inisialisasi session state
    if "bg_removed_image" not in st.session_state:
        st.session_state.bg_removed_image = None
    if "pred_class" not in st.session_state:
        st.session_state.pred_class = "-"
    if "conf_text" not in st.session_state:
        st.session_state.conf_text = "-"

    # File Uploader
    uploaded_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is None:
        # Reset jika tidak ada file
        st.session_state.bg_removed_image = None
        st.session_state.pred_class = "-"
        st.session_state.conf_text = "-"
        st.markdown("<div class='img-preview-container'><span class='img-placeholder-text'>Pilih Gambar</span></div>", unsafe_allow_html=True)
    else:
        # Tampilkan Gambar
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            # Tampilkan Gambar Asli
            buf1 = BytesIO()
            image.convert("RGB").save(buf1, format="JPEG")
            img_str1 = base64.b64encode(buf1.getvalue()).decode()
            st.markdown(f"""
            <div class='img-preview-container'>
                <img src="data:image/jpeg;base64,{img_str1}">
            </div>
            <div style='text-align:center; font-weight:600;'>Gambar Asli</div>
            """, unsafe_allow_html=True)
            
        with col2:
            # Tampilkan Hasil Remove BG
            if st.session_state.bg_removed_image is not None:
                buf2 = BytesIO()
                st.session_state.bg_removed_image.save(buf2, format="JPEG")
                img_str2 = base64.b64encode(buf2.getvalue()).decode()
                img_html = f'<img src="data:image/jpeg;base64,{img_str2}">'
            else:
                img_html = "<span class='img-placeholder-text'>Belum Diproses</span>"
                
            st.markdown(f"""
            <div class='img-preview-container'>
                {img_html}
            </div>
            <div style='text-align:center; font-weight:600;'>Hasil Hapus Background</div>
            """, unsafe_allow_html=True)

        # Tombol Aksi
        st.markdown("<div class='button-group'>", unsafe_allow_html=True)
        if st.session_state.bg_removed_image is None:
            if st.button("Hapus Latar Belakang", key="core_btn_rm"):
                with st.spinner("Menghapus background..."):
                    # Proses rembg
                    output_img = remove(image)
                    bg_white = Image.new("RGB", output_img.size, (255, 255, 255))
                    bg_white.paste(output_img, mask=output_img.split()[3])
                    st.session_state.bg_removed_image = bg_white
                st.rerun()
        else:
            c_b1, c_b2 = st.columns(2)
            with c_b1:
                if st.button("Reset", key="core_btn_reset"):
                    st.session_state.bg_removed_image = None
                    st.session_state.pred_class = "-"
                    st.session_state.conf_text = "-"
                    st.rerun()
            with c_b2:
                if st.button("Analisis", key="core_btn_anlz"):
                    if model is None:
                        st.error("Model tidak dimuat dengan benar.")
                    else:
                        with st.spinner("Sedang menganalisis..."):
                            # Preprocessing gambar untuk model
                            # Kita ambil gambar yang sudah di-remove background
                            img_proc = st.session_state.bg_removed_image.resize((224, 224))
                            img_array = np.array(img_proc)
                            img_array = np.expand_dims(img_array, axis=0)
                            
                            # Prediksi
                            # Tidak perlu bagi 255 karena model memiliki layer Lambda(preprocess_input)
                            prediction = model.predict(img_array)
                            max_conf = np.max(prediction)
                            
                            # Threshold sederhana untuk validasi
                            if max_conf < 0.40:
                                st.session_state.pred_class = "Tidak Dikenali"
                                st.session_state.conf_text = "0.00 %"
                            else:
                                st.session_state.pred_class = classes[np.argmax(prediction)]
                                st.session_state.conf_text = f"{max_conf * 100:.2f} %"
                        st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Tampilkan Hasil
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

# ==============================================================================
# 5. RENDER HALAMAN
# ==============================================================================

# Navbar
try:
    with open("logo_umrah.png", "rb") as f:
        encoded_nav_logo = base64.b64encode(f.read()).decode()
    nav_html = f'<img src="data:image/png;base64,{encoded_nav_logo}" width="30">'
except:
    nav_html = ""

st.markdown(f"""
<div class="navbar">
    {nav_html}
    <span class="navbar-title">Universitas Maritim Raja Ali Haji</span>
</div>
""", unsafe_allow_html=True)

# Welcome Screen
st.markdown("""
<div class="welcome-container">
    <div class="welcome-title">Sistem Klasifikasi Jenis Gonggong</div>
    <div class="welcome-subtitle">
        Identifikasi jenis siput Gonggong khas Kepulauan Riau menggunakan 
        teknologi Artificial Intelligence (Convolutional Neural Network - MobileNet).
    </div>
    <a class="cta-scroll-button" href="#mulai-klasifikasi">Mulai Klasifikasi</a>
</div>
<div id="mulai-klasifikasi"></div>
""", unsafe_allow_html=True)

# Header App
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
        <div class="app-title-main">Klasifikasi Jenis Gonggong</div>
        <div class="app-subtitle-main">Berbasis MobileNet Transfer Learning</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Panggil fungsi core
application_core()

# Footer
st.markdown("""
<div class='white-footer-canvas'>
    © 2026 Sistem Klasifikasi Jenis Gonggong berbasis CNN MobileNet<br>
    Fakultas Teknik dan Teknologi Kemaritiman - UMRAH
</div>
""", unsafe_allow_html=True)
