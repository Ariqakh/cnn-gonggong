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

# --- STYLING, WELCOME SECTION & ANIMASI LOADING MODERN ---
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

/* NAVBAR FIXED */
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

/* WELCOME SECTION BANNER */
.welcome-container {
    background: rgba(9, 26, 54, 0.05);
    border-left: 5px solid #091a36;
    padding: 20px;
    border-radius: 4px 16px 16px 4px;
    margin-top: 30px;
    margin-bottom: 25px;
    animation: fadeInUp 0.8s ease-out;
}
.welcome-title {
    font-size: 24px;
    font-weight: 800;
    color: #091a36;
    margin: 0 0 5px 0;
}
.welcome-text {
    font-size: 14px;
    color: #33536b;
    margin: 0;
    line-height: 1.5;
    font-weight: 500;
}

/* HEADER ANIMATION ENTRY */
.app-header {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 40px;
    margin-bottom: 12px;
    text-align: left;
    animation: fadeInUp 0.9s ease-out;
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

/* IMAGE PREVIEW EFFECT */
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
.img-preview-container:hover {
    transform: scale(1.02);
    box-shadow: 0 10px 20px rgba(11, 29, 58, 0.15);
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
    transition: background 0.3s ease;
}
[data-testid="stFileUploader"] section:hover {
    background-color: #EAEAEA !important;
}

/* WARNING BOX */
.warning-box {
    background-color: #FFDADA;
    color: #CC0000;
    padding: 15px;
    border-radius: 15px;
    font-size: 14px;
    margin-bottom: 10px;
    font-weight: 600;
    animation: shake 0.5s ease-in-out;
}

/* BUTTONS */
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
div.stButton:nth-of-type(1) > button {
    background: #0a3d3c !important;
}
div.stButton > button#btn_analyze,
div.stButton > button[key="btn_analyze"] {
    background: #115c5a !important;
}

/* RESULTS CARDS */
.result-box {
    background-color: #87D4D4;
    border-radius: 20px;
    padding: 20px 25px;
    margin-top: 20px;
    text-align: left;
    display: flex;
    justify-content: flex-start;
    align-items: center;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    animation: fadeInUp 0.6s ease-out both;
}

/* ===============================
   ANIMASI LOADING BARU & RAPI (PREMIUM LOOK)
   =============================== */
.premium-loader-card {
    background: #ffffff;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(9, 26, 54, 0.08);
    margin: 20px auto;
    max-width: 450px;
    animation: fadeIn 0.4s ease-in-out;
}

.wave-loading-dots {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 8px;
    margin-bottom: 15px;
}

.wave-loading-dots span {
    display: inline-block;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #115c5a;
    animation: wavePulse 1.2s infinite ease-in-out;
}

.wave-loading-dots span:nth-child(2) { animation-delay: 0.15s; background: #0a3d3c; }
.wave-loading-dots span:nth-child(3) { animation-delay: 0.3s; background: #43647d; }
.wave-loading-dots span:nth-child(4) { animation-delay: 0.45s; background: #87D4D4; }

.loader-status-text {
    font-size: 16px;
    font-weight: 700;
    color: #0b1d3a;
    margin: 5px 0;
    letter-spacing: 0.3px;
}

.loader-sub-text {
    font-size: 12px;
    font-weight: 500;
    color: #728da1;
}

/* KEYFRAMES CORE */
@keyframes wavePulse {
    0%, 100% { transform: translateY(0) scale(1); opacity: 0.5; }
    50% { transform: translateY(-10px) scale(1.1); opacity: 1; }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes shake {
    0%, 100% { transform: translateX(0); }
    20%, 60% { transform: translateX(-6px); }
    40%, 80% { transform: translateX(6px); }
}

.result-box-spacer { height: 100px; width: 100%; }
.page-wrapper { display: flex !important; flex-direction: column !important; flex-grow: 1 !important; min-height: 100% !important; }
.white-footer-canvas { position: relative !important; margin-top: auto !important; padding: 20px 0px !important; display: flex !important; justify-content: center !important; align-items: center !important; width: 100% !important; }
.footer-text { text-align: center; color: #1a364a; font-size: 14px; font-weight: 500; line-height: 1.5; margin: 0 auto; }

/* RESPONSIVE MOBILE */
@media (max-width: 480px) {
    .navbar { padding: 15px 15px; gap: 8px; }
    .navbar-title { font-size: 14px; }
    .welcome-container { margin-top: 15px; padding: 15px; }
    .welcome-title { font-size: 18px; }
    .welcome-text { font-size: 12px; }
    .app-header { flex-direction: column; gap: 10px; text-align: center; }
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

# --- MODEL LOADING ---
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
    'Canarium_Mutabile', 
    'Canarium_Urseus', 
    'Laevistrombus_Turturella', 
    'Pugilina_Coclidium'
]

# --- STATIC NAVBAR ---
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

# --- NEW: WELCOME BANNER SECTION ---
st.markdown("""
<div class="welcome-container">
    <div class="welcome-title">Selamat Datang di Portal Klasifikasi! 👋</div>
    <div class="welcome-text">
        Sistem cerdas berbasis kecerdasan buatan (AI) dirancang untuk mempermudah Anda mengidentifikasi spesies biota laut Gonggong khas Kepulauan Riau secara cepat, presisi, dan otomatis.
    </div>
</div>
""", unsafe_allow_html=True)

# --- MAIN HEADER ---
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

# --- APPLICATION DESKTOP WORKSPACE (FRAGMENT) ---
@st.fragment
def main_workspace():
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
            <span class='img-placeholder-text'>Silakan Upload Gambar Terlebih Dahulu</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        image = Image.open(uploaded_file)
        
        # Grid Display Preview Gambar
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

        # Kontrol Button Group
        st.markdown("<div class='button-group'>", unsafe_allow_html=True)
        if st.session_state.bg_removed_image is None:
            if st.button("Hapus Latar Belakang", key="frag_btn_rm"):
                # Menampilkan loader premium gelombang ombak
                st.markdown("""
                <div class="premium-loader-card">
                    <div class="wave-loading-dots">
                        <span></span><span></span><span></span><span></span>
                    </div>
                    <div class="loader-status-text">AI Segmentasi Aktif</div>
                    <div class="loader-sub-text">Mengekstraksi objek gonggong dari latar belakang...</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Olah segmentasi AI rembg
                output_img = remove(image)
                bg_white = Image.new("RGB", output_img.size, (255, 255, 255))
                bg_white.paste(output_img, mask=output_img.split()[3])
                
                st.session_state.bg_removed_image = bg_white
                st.rerun()
        else:
            c_b1, c_b2 = st.columns(2)
            with c_b1:
                if st.button("Proses Ulang Gambar", key="frag_btn_reset"):
                    st.session_state.bg_removed_image = None
                    st.session_state.pred_class = "-"
                    st.session_state.conf_text = "-"
                    st.session_state.warn_html = ""
                    st.rerun()
            with c_b2:
                if st.button("Analisis Gambar", key="frag_btn_anlz"):
                    # Menampilkan loader premium gelombang ombak untuk proses klasifikasi
                    st.markdown("""
                    <div class="premium-loader-card">
                        <div class="wave-loading-dots">
                            <span></span><span></span><span></span><span></span>
                        </div>
                        <div class="loader-status-text">Mengevaluasi Gambar</div>
                        <div class="loader-sub-text">Menganalisis fitur morfologi cangkang gonggong...</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    segmented_np = np.array(st.session_state.bg_removed_image)
                    
                    # Sharpening via OpenCV
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
                        st.session_state.warn_html = "<div class='warning-box'>⚠️ Gambar tidak dikenali sebagai Gonggong. Harap upload foto Gonggong yang lebih jelas.</div>"
                        st.session_state.pred_class = "-"
                        st.session_state.conf_text = "-"
                    else:
                        st.session_state.warn_html = ""
                        st.session_state.pred_class = classes[np.argmax(prediction)].replace("_", " ")
                        st.session_state.conf_text = f"{max_conf * 100:.2f} %"
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Hasil Analisis Akhir
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

main_workspace()

st.markdown("</div>", unsafe_allow_html=True) 

# --- FOOTER ---
st.markdown("""
<div class='white-footer-canvas'>
    <div class='footer-text'>
        © 2026 Sistem Klasifikasi Jenis Gonggong berbasis CNN MobileNet<br>
        Fakultas Teknik dan Teknologi Kemaritiman - UMRAH
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
