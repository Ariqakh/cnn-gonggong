import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import base64
from io import BytesIO
from rembg import remove
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input

st.set_page_config(page_title="Klasifikasi Jenis Gonggong", layout="centered")

# --- CSS (TIDAK DIUBAH) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { font-family: 'Inter', sans-serif; margin: 0; padding: 0; height: 100%; }
[data-testid="stAppViewContainer"] { background: linear-gradient(180deg, #A7FFFF 0%, #D1FFFF 100%) !important; min-height: 100vh !important; display: flex !important; flex-direction: column !important; }
[data-testid="stMain"] { background: transparent !important; padding-bottom: 0px !important; display: flex !important; flex-direction: column !important; flex-grow: 1 !important; }
[data-testid="stMainBlockContainer"] { display: flex !important; flex-direction: column !important; flex-grow: 1 !important; padding-bottom: 0px !important; }
header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; display: none !important; }
.navbar { position: fixed; top: 0; left: 0; right: 0; z-index: 9999; background: #091a36; padding: 25px 20px; display: flex; align-items: center; gap: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
.navbar-title { color: #ffffff; font-size: 20px; font-weight: 600; }
.welcome-container { display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; min-height: 70vh; padding: 60px 20px 20px 20px; animation: fadeIn 1s ease-out; }
.welcome-title { font-size: 56px; font-weight: 800; color: #0b1d3a; line-height: 1.1; margin-bottom: 15px; text-transform: uppercase; }
.welcome-subtitle { font-size: 18px; color: #43647d; font-weight: 500; max-width: 600px; margin-bottom: 40px; line-height: 1.6; }
.cta-scroll-button { background: #0a3d3c; color: white !important; padding: 16px 36px; border-radius: 35px; font-size: 16px; font-weight: 700; text-decoration: none !important; box-shadow: 0 10px 25px rgba(10, 61, 60, 0.3); transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1); display: inline-block; }
.cta-scroll-button:hover { transform: translateY(-4px); box-shadow: 0 15px 30px rgba(10, 61, 60, 0.4); background: #115c5a; }
.app-header { display: flex; align-items: center; justify-content: center; gap: 40px; margin: 60px 0 12px 0; text-align: left; padding-top: 40px; }
.app-logo-img { width: 370px; height: 300px; }
.app-title-container { display: flex; flex-direction: column; }
.app-title-main { font-size: 65px; font-weight: 800; color: #0b1d3a; line-height: 1.05; text-transform: uppercase; letter-spacing: 1px; }
.app-subtitle-main { font-size: 15px; color: #43647d; font-weight: 600; margin-top: 6px; }
.img-preview-container { background: #E8E8E8; border-radius: 20px; margin: 15px auto; width: 100%; height: 260px; display: flex; align-items: center; justify-content: center; overflow: hidden; border: 1px solid #ddd; }
.img-preview-container img { max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 10px; }
.img-placeholder-text { font-size: 16px; color: #666; font-weight: 500; }
[data-testid="stFileUploader"] section { background-color: #F3F3F3 !important; border: 1px solid #ccc !important; border-radius: 30px !important; padding: 15px !important; }
.button-group { display: flex; flex-direction: row; justify-content: center; gap: 15px; margin: 20px auto; width: 100%; }
div.stButton > button { color: white !important; font-size: 16px !important; font-weight: 700 !important; padding: 15px 25px !important; border-radius: 30px !important; border: none !important; width: 100%; box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important; }
div[data-testid="stButton"]:has(button[kind="secondary"]) > button, div.stButton > button[kind="secondary"], div.stButton:nth-of-type(1) > button { background: #0a3d3c !important; }
div.stButton > button[key*="anlz"] { background: #115c5a !important; }
div.stButton > button[key*="reset"] { background: #64748b !important; }
.result-box { background-color: #87D4D4; border-radius: 20px; padding: 20px 25px; margin-top: 20px; text-align: left; display: flex; align-items: center; }
.result-label { font-weight: 700; color: #0b1d3a; font-size: 18px; min-width: 180px; }
.result-value { font-weight: 700; color: #0b1d3a; font-size: 18px; }
.warning-box { background-color: #FFDADA; color: #CC0000; padding: 10px; border-radius: 15px; font-size: 13px; margin-bottom: 10px; font-weight: 600; }
.result-box-spacer { height: 100px; width: 100%; }
.page-wrapper { display: flex !important; flex-direction: column !important; flex-grow: 1 !important; min-height: 100% !important; }
.white-footer-canvas { position: relative !important; margin-top: auto !important; padding: 20px 0px !important; display: flex !important; justify-content: center !important; }
.footer-text { text-align: center; color: #1a364a; font-size: 14px; font-weight: 500; }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_my_model():
    import keras
    # Mendaftarkan preprocess_input sebagai objek yang dikenal
    @keras.saving.register_keras_serializable()
    def custom_preprocess(x): return preprocess_input(x)
    return load_model("model_gonggong.h5", custom_objects={'preprocess_input': custom_preprocess}, compile=False)

model = load_my_model()
class_names = ['Canarium Mutabile', 'Canarium Urseus', 'Laevistrombus Turturella', 'Pugilina Coclidium']

# --- UI & LOGIC ---
st.markdown("<div class='page-wrapper'>", unsafe_allow_html=True)
st.markdown("<div class='welcome-container'><div class='welcome-title'>Selamat Datang di<br>Sistem Klasifikasi Gonggong</div><a class='cta-scroll-button' href='#mulai-klasifikasi'>Klasifikasi Sekarang</a></div><div id='mulai-klasifikasi'></div>", unsafe_allow_html=True)

@st.fragment
def application_core():
    if "bg_removed_image" not in st.session_state: st.session_state.bg_removed_image = None
    if "pred_class" not in st.session_state: st.session_state.pred_class = "-"
    if "conf_text" not in st.session_state: st.session_state.conf_text = "-"
    if "warn_html" not in st.session_state: st.session_state.warn_html = ""

    uploaded_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is None:
        st.session_state.bg_removed_image = None
        st.markdown("<div class='img-preview-container'><span class='img-placeholder-text'>Gambar</span></div>", unsafe_allow_html=True)
    else:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        with col1: st.image(image, caption="Gambar Asli", use_column_width=True)
        with col2:
            if st.session_state.bg_removed_image: st.image(st.session_state.bg_removed_image, caption="Hasil Background Removed", use_column_width=True)
            else: st.info("Belum diproses")

        st.markdown("<div class='button-group'>", unsafe_allow_html=True)
        if st.session_state.bg_removed_image is None:
            if st.button("Hapus Latar Belakang", key="core_btn_rm"):
                output = remove(image)
                bg_white = Image.new("RGB", output.size, (255, 255, 255))
                bg_white.paste(output, mask=output.split()[3])
                st.session_state.bg_removed_image = bg_white
                st.rerun()
        else:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Ulang", key="core_btn_reset"):
                    st.session_state.bg_removed_image = None; st.rerun()
            with c2:
                if st.button("Analisis Gambar", key="core_btn_anlz"):
                    # SHARPENING + PREPROCESSING
                    img_np = np.array(st.session_state.bg_removed_image)
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                    sharpened = cv2.filter2D(img_bgr, -1, kernel)
                    
                    final_img = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)).resize((224, 224))
                    img_array = preprocess_input(np.expand_dims(np.array(final_img), axis=0))

                    prediction = model.predict(img_array)
                    idx = np.argmax(prediction[0])
                    max_conf = prediction[0][idx]
                    
                    st.session_state.pred_class = class_names[idx]
                    st.session_state.conf_text = f"{max_conf * 100:.2f} %"
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='result-box'><span class='result-label'>Jenis Gonggong :</span><span class='result-value'>{st.session_state.pred_class}</span></div>
    <div class='result-box'><span class='result-label'>Tingkat Akurasi :</span><span class='result-value'>{st.session_state.conf_text}</span></div>
    """, unsafe_allow_html=True)

application_core()
st.markdown("</div>", unsafe_allow_html=True)
