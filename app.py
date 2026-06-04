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

# --- SEMUA STYLING, ANIMASI & DESAIN AESTHETIC ---
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
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #A7FFFF 0%, #D1FFFF 100%) !important;
    min-height: 100vh !important;
}

[data-testid="stMain"] {
    background: transparent !important;
    padding-bottom: 0px !important; 
}

header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { 
    visibility: hidden; display: none !important; 
}

/* NAVBAR FIXED */
.navbar {
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 9999;
    background: #091a36;
    padding: 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}
.navbar-title {
    color: #ffffff;
    font-size: 16px;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* WELCOME HERO SECTION */
.welcome-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    min-height: 85vh;
    padding: 100px 20px 40px 20px;
    animation: fadeIn 1s ease-out;
}

.welcome-badge {
    background: rgba(11, 29, 58, 0.1);
    color: #0b1d3a;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 20px;
    letter-spacing: 1px;
    text-transform: uppercase;
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

/* BUTTON JUMP TO CLASSIFICATION (AESTHETIC) */
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

/* APP WORKSPACE HEADER */
.app-header {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 30px;
    margin: 60px 0 30px 0;
    text-align: left;
    padding-top: 40px;
}
.app-logo-img {
    width: 280px;
    height: auto;
}
.app-title-main {
    font-size: 40px; 
    font-weight: 800;
    color: #0b1d3a;
    line-height: 1.1;
    text-transform: uppercase;
}

/* MAIN CARD CONTAINER */
.main-card {
    background: rgba(255, 255, 255, 0.4);
    backdrop-filter: blur(10px);
    border-radius: 30px;
    padding: 30px;
    border: 1px solid rgba(255,255,255,0.5);
    box-shadow: 0 15px 35px rgba(0,0,0,0.05);
    margin-bottom: 40px;
}

/* IMAGE PREVIEW BOX WITH SMOOTH SHADOW */
.img-preview-container {
    background: #F0F2F5;
    border-radius: 20px;
    margin: 15px auto;
    width: 100%;
    height: 260px; 
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    border: 2px dashed #cbd5e1;
    transition: all 0.3s ease;
}
.img-preview-container img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    animation: fadeIn 0.4s ease-in-out;
}

.img-placeholder-text {
    font-size: 15px;
    color: #64748b;
    font-weight: 500;
}

[data-testid="stFileUploader"] section {
    background-color: #FFFFFF !important;
    border: 2px dashed #cbd5e1 !important;
    border-radius: 20px !important;
    padding: 20px !important;
}

/* BUTTONS */
.button-group {
    display: flex;
    gap: 15px;
    margin: 25px 0;
}
div.stButton > button {
    color: white !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    padding: 14px 28px !important;
    border-radius: 25px !important;
    border: none !important;
    width: 100%;
    transition: all 0.3s ease !important;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1) !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(0,0,0,0.18) !important;
}
div.stButton > button[key*="rm"], div.stButton:nth-of-type(1) > button { background: #0a3d3c !important; }
div.stButton > button[key*="anlz"] { background: #115c5a !important; }
div.stButton > button[key*="reset"] { background: #64748b !important; }

/* REFINED HIGH-END LOADING ANIMATION */
.premium-loader {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 30px;
    background: rgba(255, 255, 255, 0.9);
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    margin: 20px 0;
    animation: fadeInUp 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
}
.loader-spinner {
    width: 50px;
    height: 50px;
    border: 3px solid #e2e8f0;
    border-top: 3px solid #0a3d3c;
    border-radius: 50%;
    animation: spin 0.8s cubic-bezier(0.55, 0.055, 0.675, 0.19) infinite;
}
.loader-text {
    margin-top: 15px;
    font-size: 15px;
    font-weight: 600;
    color: #0b1d3a;
    letter-spacing: 0.3px;
}

/* RESULTS BOX */
.result-box {
    background: #ffffff;
    border-left: 6px solid #115c5a;
    border-radius: 15px;
    padding: 18px 25px;
    margin-top: 15px;
    display: flex;
    align-items: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.03);
}
.result-label {
    font-weight: 700;
    color: #475569;
    font-size: 16px;
    min-width: 160px;
}
.result-value {
    font-weight: 800;
    color: #0b1d3a;
    font-size: 18px;
}

.warning-box {
    background-color: #fef2f2;
    color: #dc2626;
    padding: 15px;
    border-radius: 15px;
    border-left: 5px solid #dc2626;
    font-weight: 600;
    margin: 15px 0;
}

/* KEYFRAMES */
@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes fadeInUp { from { opacity: 0; transform: translateY(15px); } to { opacity: 1; transform: translateY(0); } }

/* RESPONSIVE MOBILE */
@media (max-width: 480px) {
    .navbar-title { font-size: 13px; }
    .welcome-title { font-size: 34px; }
    .welcome-subtitle { font-size: 14px; }
    .app-header { flex-direction: column; text-align: center; gap: 15px; }
    .app-logo-img { width: 180px; }
    .app-title-main { font-size: 26px; }
    .button-group { flex-direction: column; }
    .img-preview-container { height: 180px; }
}
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING (CACHED) ---
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

    return keras_load_model("model_gonggong.h5", compile=False)

model = load_my_model()

classes = [
    'Canarium Mutabile', 
    'Canarium Urseus', 
    'Laevistrombus Turturella', 
    'Pugilina Coclidium'
]

# --- NAVBAR ---
try:
    with open("logo_umrah.png", "rb") as f:
        encoded_nav_logo = base64.b64encode(f.read()).decode()
except:
    encoded_nav_logo = ""

st.markdown(f"""
<div class="navbar">
    <div class="navbar-logo">
        <img src="data:image/png;base64,{encoded_nav_logo}" width="28" height="28" style="object-fit:cover;">
    </div>
    <span class="navbar-title">Universitas Maritim Raja Ali Haji</span>
</div>
""", unsafe_allow_html=True)

# --- 1. WELCOME HERO SECTION ---
st.markdown("""
<div class="welcome-container">
    <div class="welcome-badge">Deep Learning Project</div>
    <div class="welcome-title">Selamat Datang di<br>Sistem Klasifikasi Gonggong</div>
    <div class="welcome-subtitle">Identifikasi jenis siput Gonggong khas Kepulauan Riau secara instan dan akurat menggunakan teknologi Artificial Intelligence berbasis Convolutional Neural Network (MobileNet).</div>
    <a class="cta-scroll-button" href="#mulai-klasifikasi">Klasifikasi Sekarang</a>
</div>
<div id="mulai-klasifikasi"></div>
""", unsafe_allow_html=True)

# --- 2. WORKSPACE HEADER ---
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
        <div class="app-title-main">Mulai Analisis<br>Gambar Citra</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='main-card'>", unsafe_allow_html=True)

# --- 3. ISOLATED APP SYSTEM (FRAGMENT FOR ANTI-FLICKER) ---
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
            <span class='img-placeholder-text'>Silakan seret atau pilih file foto Gonggong</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        image = Image.open(uploaded_file)
        
        # Grid Tampilan Gambar Preview
        col1, col2 = st.columns(2)
        with col1:
            buf1 = BytesIO()
            image.convert("RGB").save(buf1, format="JPEG")
            img_str1 = base64.b64encode(buf1.getvalue()).decode()
            st.markdown(f"""
            <div class='img-preview-container'>
                <img src="data:image/jpeg;base64,{img_str1}">
            </div>
            <div style='text-align:center; font-weight:600; color:#475569; font-size:13px;'>Gambar Asli</div>
            """, unsafe_allow_html=True)
            
        with col2:
            if st.session_state.bg_removed_image is not None:
                buf2 = BytesIO()
                st.session_state.bg_removed_image.save(buf2, format="JPEG")
                img_str2 = base64.b64encode(buf2.getvalue()).decode()
                img_html = f'<img src="data:image/jpeg;base64,{img_str2}">'
            else:
                img_html = "<span class='img-placeholder-text'>Menunggu Proses Background</span>"
                
            st.markdown(f"""
            <div class='img-preview-container'>
                {img_html}
            </div>
            <div style='text-align:center; font-weight:600; color:#475569; font-size:13px;'>Hasil Segmentasi AI</div>
            """, unsafe_allow_html=True)

        # Logika Alur Tombol Kontrol
        st.markdown("<div class='button-group'>", unsafe_allow_html=True)
        if st.session_state.bg_removed_image is None:
            if st.button("Hapus Latar Belakang", key="core_btn_rm"):
                # Efek Loading Premium muncul di posisi tombol
                st.markdown("""
                <div class='premium-loader'>
                    <div class='loader-spinner'></div>
                    <div class='loader-text'>AI sedang memotong & membersihkan background...</div>
                </div>
                """, unsafe_allow_html=True)
                
                output_img = remove(image)
                bg_white = Image.new("RGB", output_img.size, (255, 255, 255))
                bg_white.paste(output_img, mask=output_img.split()[3])
                
                st.session_state.bg_removed_image = bg_white
                st.rerun()
        else:
            c_b1, c_b2 = st.columns(2)
            with c_b1:
                if st.button("Reset / Upload Ulang", key="core_btn_reset"):
                    st.session_state.bg_removed_image = None
                    st.session_state.pred_class = "-"
                    st.session_state.conf_text = "-"
                    st.session_state.warn_html = ""
                    st.rerun()
            with c_b2:
                if st.button("Mulai Analisis Gambar", key="core_btn_anlz"):
                    st.markdown("""
                    <div class='premium-loader'>
                        <div class='loader-spinner'></div>
                        <div class='loader-text'>Mengekstrak fitur citra & melakukan klasifikasi...</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    segmented_np = np.array(st.session_state.bg_removed_image)
                    
                    # Pemrosesan Perbaikan Citra (Sharpening)
                    img_bgr = cv2.cvtColor(segmented_np, cv2.COLOR_RGB2BGR)
                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                    sharpened = cv2.filter2D(img_bgr, -1, kernel)
                    final_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
                    
                    final_processed = Image.fromarray(final_rgb)
                    img_resized = final_processed.resize((224, 224))
                    img_array = np.array(img_resized).astype("float32") / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    prediction = model.predict(img_array)
                    max_conf = np.max(prediction)
                    
                    pure_white = np.sum(np.all(segmented_np >= 245, axis=-1))
                    total_pixels = segmented_np.shape[0] * segmented_np.shape[1]
                    
                    if max_conf < 0.50 or (pure_white / total_pixels) > 0.95:
                        st.session_state.warn_html = "<div class='warning-box'>⚠️ Citra objek gagal dikenali. Pastikan Anda mengunggah foto Gonggong yang utuh dan jelas.</div>"
                        st.session_state.pred_class = "-"
                        st.session_state.conf_text = "-"
                    else:
                        st.session_state.warn_html = ""
                        st.session_state.pred_class = classes[np.argmax(prediction)]
                        st.session_state.conf_text = f"{max_conf * 100:.2f} %"
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Tampilan Hasil Eksekusi Model
    if st.session_state.warn_html:
        st.markdown(st.session_state.warn_html, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='result-box' style='animation: fadeInUp 0.4s ease-out forwards; animation-delay: 0.1s;'>
        <span class='result-label'>Jenis Gonggong</span>
        <span class='result-value'>: &nbsp;{st.session_state.pred_class}</span>
    </div>
    <div class='result-box' style='animation: fadeInUp 0.4s ease-out forwards; animation-delay: 0.2s;'>
        <span class='result-label'>Tingkat Akurasi</span>
        <span class='result-value'>: &nbsp;{st.session_state.conf_text}</span>
    </div>
    """, unsafe_allow_html=True)

# Eksekusi sistem fragment inti
application_core()

st.markdown("</div>", unsafe_allow_html=True) # End Main Card

# --- 4. FOOTER REGION ---
st.markdown("""
<div style="height: 60px;"></div>
<div class='white-footer-canvas'>
    <div class='footer-text'>
        © 2026 Sistem Klasifikasi Jenis Gonggong berbasis CNN MobileNet<br>
        Fakultas Teknik dan Teknologi Kemaritiman - UMRAH
    </div>
</div>
""", unsafe_allow_html=True)
