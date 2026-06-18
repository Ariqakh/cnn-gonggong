import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
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

html {
    scroll-behavior: smooth;
}

body {
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

.navbar {
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 9999;
    background: #091a36;
    padding: 25px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}
.navbar-title {
    color: #ffffff;
    font-size: 20px;
    font-weight: 600;
}

.welcome-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    min-height: 70vh;
    padding: 60px 20px 20px 20px;
    animation: fadeIn 1s ease-out;
}

.welcome-title {
    font-size: 56px;
    font-weight: 800;
    color: #0b1d3a;
    line-height: 1.1;
    margin-bottom: 15px;
    text-transform: uppercase;
}

.welcome-subtitle {
    font-size: 18px;
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

.app-header {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 40px;
    margin: 60px 0 12px 0;
    text-align: left;
    padding-top: 40px;
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
    transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
}
.img-preview-container img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    border-radius: 10px;
    animation: fadeIn 0.5s ease-in-out;
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
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
}
div.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 20px rgba(0,0,0,0.25) !important;
    opacity: 0.95 !important;
}
div[data-testid="stButton"]:has(button[kind="secondary"]) > button,
div.stButton > button[kind="secondary"],
div.stButton:nth-of-type(1) > button { background: #0a3d3c !important; }
div.stButton > button[key*="anlz"] { background: #115c5a !important; }
div.stButton > button[key*="reset"] { background: #64748b !important; }
div[data-testid="stSpinner"] {
    text-align: center !important;
}

div[data-testid="stSpinner"] > div {
    border-top-color: #0a3d3c !important;
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
    animation: fadeInUp 0.6s ease-out both;
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

.result-box-spacer { height: 100px; width: 100%; }
.page-wrapper { display: flex !important; flex-direction: column !important; flex-grow: 1 !important; min-height: 100% !important; }
.white-footer-canvas { position: relative !important; margin-top: auto !important; padding: 20px 0px !important; display: flex !important; justify-content: center !important; align-items: center !important; width: 100% !important; }
.footer-text { text-align: center; color: #1a364a; font-size: 14px; font-weight: 500; line-height: 1.5; margin: 0 auto; }

@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

@media (max-width: 480px) {
    .navbar { padding: 15px 15px; gap: 8px; }
    .navbar-title { font-size: 14px; }
    .welcome-title { font-size: 32px; }
    .welcome-subtitle { font-size: 12px; }
    .app-header { flex-direction: column; gap: 10px; text-align: center; margin-top: -10px; }
    .app-logo-img { width: 195px; height: auto; }
    .app-title-main { font-size: 32px; }
    .img-preview-container { height: 160px; margin: 8px auto; }
    .button-group { flex-direction: column !important; gap: 10px; }
    div.stButton > button { width: 100% !important; font-size: 15px; padding: 12px 20px !important; }
    .result-box { padding: 15px 20px; display: flex !important; flex-direction: row !important; }
    .result-label { font-size: 15px; min-width: 135px; }
    .result-value { font-size: 15px; }
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_my_model():
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

    # Integrasi Kunci Kesembuhan: Daftarkan preprocess_input ke custom_objects
    custom_objects = {'preprocess_input': preprocess_input}
    return keras_load_model("model_gonggong.h5", custom_objects=custom_objects, compile=False)

model = load_my_model()

classes = [
    'Canarium Mutabile', 
    'Canarium Urseus', 
    'Laevistrombus Turturella', 
    'Pugilina Coclidium'
]

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
""", unsafe_allow_html=True)

st.markdown("<div class='page-wrapper'>", unsafe_allow_html=True)

st.markdown("""
<div class="welcome-container">
    <div class="welcome-title">Selamat Datang di<br>Sistem Klasifikasi Gonggong</div>
    <div class="welcome-subtitle">Identifikasi jenis siput Gonggong khas Kepulauan Riau secara instan dan akurat menggunakan teknologi Artificial Intelligence berbasis Convolutional Neural Network (MobileNet).</div>
    <a class="cta-scroll-button" href="#mulai-klasifikasi">Klasifikasi Sekarang</a>
</div>
<div id="mulai-klasifikasi"></div>
""", unsafe_allow_html=True)

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

@st.fragment
def application_core():
    if "bg_removed_image" not in st.session_state:
        st.session_state.bg_removed_image = None
    if "pred_class" not in st.session_state:
        st.session_state.pred_class = "-"
    if "conf_text" not in st.session_state:
        st.session_state.conf_text = "-"
    if "warn_html" not in st.session_state:
        st.session_state.warn_html = ""

    uploaded_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is None:
        st.session_state.bg_removed_image = None
        st.session_state.pred_class = "-"
        st.session_state.conf_text = "-"
        st.session_state.warn_html = ""
        
        st.markdown("""
        <div class='img-preview-container'>
            <span class='img-placeholder-text'>Gambar</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            buf1 = BytesIO()
            image.convert("RGB").save(buf1, format="JPEG")
            img_str1 = base64.b64encode(buf1.getvalue()).decode()
            st.markdown(f"""
            <div class='img-preview-container'>
                <img src="data:image/jpeg;base64,{img_str1}">
            </div>
            <div style='text-align:center; font-weight:600; color:#0b1d3a; font-size:13px;'>Gambar Asli</div>
            """, unsafe_allow_html=True)
            
        with col2:
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
            <div style='text-align:center; font-weight:600; color:#0b1d3a; font-size:13px;'>Hasil Hapus Background</div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='button-group'>", unsafe_allow_html=True)
        if st.session_state.bg_removed_image is None:
            # SOLUSI 2: Menyediakan dua tombol agar user bisa langsung menganalisis gambar asli jika rembg bermasalah
            c_init1, c_init2 = st.columns(2)
            with c_init1:
                if st.button("Hapus Latar Belakang", key="core_btn_rm"):
                    with st.spinner(""):
                        output_img = remove(image)
                        bg_white = Image.new("RGB", output_img.size, (255, 255, 255))
                        bg_white.paste(output_img, mask=output_img.split()[3])
                        st.session_state.bg_removed_image = bg_white
                    st.rerun()
            with c_init2:
                if st.button("Analisis Gambar Asli", key="core_btn_anlz_raw"):
                    with st.spinner(""):
                        # SOLUSI 1 & 2: Tanpa Filter Ketajaman OpenCV & Menggunakan Gambar Asli secara Utuh
                        img_raw_rgb = image.convert('RGB')
                        img_resized = img_raw_rgb.resize((224, 224))
                        img_array = np.array(img_resized).astype("float32")
                        
                        img_preprocessed = preprocess_input(img_array)
                        img_tensor = np.expand_dims(img_preprocessed, axis=0)
                
                        prediction = model.predict(img_tensor, verbose=0)
                        probabilities = prediction[0]
                        max_conf = np.max(probabilities)
                        
                        if max_conf < 0.50:
                            st.session_state.warn_html = "<div class='warning-box'>⚠️ Gambar tidak dikenali sebagai Gonggong. Harap upload foto Gonggong yang jelas.</div>"
                            st.session_state.pred_class = "-"
                            st.session_state.conf_text = "-"
                        else:
                            st.session_state.warn_html = ""
                            st.session_state.pred_class = classes[np.argmax(probabilities)]
                            st.session_state.conf_text = f"{max_conf * 100:.2f} %"
                    st.rerun()
        else:
            c_b1, c_b2 = st.columns(2)
            with c_b1:
                if st.button("Proses Ulang Gambar", key="core_btn_reset"):
                    st.session_state.bg_removed_image = None
                    st.session_state.pred_class = "-"
                    st.session_state.conf_text = "-"
                    st.session_state.warn_html = ""
                    st.rerun()
            with c_b2:
                if st.button("Analisis Gambar", key="core_btn_anlz"):
                    with st.spinner(""):
                        segmented_np = np.array(st.session_state.bg_removed_image)
                
                        # SOLUSI 1: Menghapus total modul kernel filter2D OpenCV yang mengacaukan ekstraksi fitur warna asli.
                        # Gambar diambil langsung murni dari konversi RGB hasil rembg
                        final_processed = Image.fromarray(segmented_np)
                        img_resized = final_processed.resize((224, 224))
                        img_array = np.array(img_resized).astype("float32")
                        
                        img_preprocessed = preprocess_input(img_array)
                        img_tensor = np.expand_dims(img_preprocessed, axis=0)
                
                        prediction = model.predict(img_tensor, verbose=0)
                        probabilities = prediction[0]
                        max_conf = np.max(probabilities)
                    
                        pure_white = np.sum(np.all(segmented_np >= 245, axis=-1))
                        total_pixels = segmented_np.shape[0] * segmented_np.shape[1]
                        
                        if max_conf < 0.50 or (pure_white / total_pixels) > 0.95:
                            st.session_state.warn_html = "<div class='warning-box'>⚠️ Gambar tidak dikenali sebagai Gonggong. Harap upload foto Gonggong yang jelas.</div>"
                            st.session_state.pred_class = "-"
                            st.session_state.conf_text = "-"
                        else:
                            st.session_state.warn_html = ""
                            st.session_state.pred_class = classes[np.argmax(probabilities)]
                            st.session_state.conf_text = f"{max_conf * 100:.2f} %"
                        st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.warn_html:
        st.markdown(st.session_state.warn_html, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='result-box' style='animation-delay: 0.1s;'>
        <span class='result-label'>Jenis Gonggong :</span>
        <span class='result-value'>{st.session_state.pred_class}</span>
    </div>
    <div class='result-box' style='animation-delay: 0.2s;'>
        <span class='result-label'>Tingkat Akurasi :</span>
        <span class='result-value'>{st.session_state.conf_text}</span>
    </div>
    <div class='result-box-spacer'></div>
    """, unsafe_allow_html=True)

application_core()

st.markdown("</div>", unsafe_allow_html=True) 

st.markdown("""
<div class='white-footer-canvas'>
    <div class='footer-text'>
        © 2026 Sistem Klasifikasi Jenis Gonggong berbasis CNN MobileNet<br>
        Fakultas Teknik dan Teknologi Kemaritiman - UMRAH
    </div>
</div>
</div>
""", unsafe_allow_html=True)
