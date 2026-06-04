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
    .img-preview-container { height: 160px; margin: 8px auto; }
    .button-group { flex-direction: column !important; gap: 10px; margin: 10px auto !important; }
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

    model = keras_load_model("model_gonggong.h5", compile=False)
    return model

model = load_my_model()

classes = [
    "Canarium Mutabile",
    "Canarium Urseus",
    "Laevistrombus Turturella",
    "Pugilina Coclidium"
]

# --- SESSION STATE ---
if "pred_class" not in st.session_state: st.session_state.pred_class = "-"
if "conf_text" not in st.session_state: st.session_state.conf_text = "-"
if "warn_box_html" not in st.session_state: st.session_state.warn_box_html = ""
if "is_bg_removed" not in st.session_state: st.session_state.is_bg_removed = False

st.markdown("<div class='page-wrapper'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    gray_np = 0.2989 * img_np[:,:,0] + 0.5870 * img_np[:,:,1] + 0.1140 * img_np[:,:,2]
    background_mask = gray_np < 40 
    segmented_np = img_np.copy()
    segmented_np[background_mask] = [255, 255, 255]
    processed_image = Image.fromarray(segmented_np)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div class='img-preview-container'>", unsafe_allow_html=True)
        st.image(image)
        st.markdown("</div><div style='text-align:center; font-weight:600;'>Gambar Asli</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='img-preview-container'>", unsafe_allow_html=True)
        if st.session_state.is_bg_removed: st.image(processed_image)
        else: st.write("Belum Diproses")
        st.markdown("</div><div style='text-align:center; font-weight:600;'>Hasil Hapus Background</div>", unsafe_allow_html=True)

    st.markdown("<div class='button-group'>", unsafe_allow_html=True)
    bg_clicked = st.button("Hapus Latar Belakang", key="btn_remove_bg")
    analyze_clicked = st.button("Analisis Gambar", key="btn_analyze")
    st.markdown("</div>", unsafe_allow_html=True)

    if bg_clicked:
        st.session_state.is_bg_removed = True
        st.rerun()

    if analyze_clicked:
        final_img = processed_image if st.session_state.is_bg_removed else image
        img_resized = final_img.resize((224, 224))
        img_array = np.expand_dims(np.array(img_resized).astype("float32") / 255.0, axis=0)
        prediction = model.predict(img_array)
        max_conf = np.max(prediction)
        
        if max_conf < 0.50:
            st.session_state.warn_box_html = "<div class='warning-box'>⚠️ Gambar tidak dikenali.</div>"
            st.session_state.pred_class = "-"
            st.session_state.conf_text = "-"
        else:
            st.session_state.pred_class = classes[np.argmax(prediction)]
            st.session_state.conf_text = f"{max_conf * 100:.2f}%"

if st.session_state.warn_box_html: st.markdown(st.session_state.warn_box_html, unsafe_allow_html=True)

st.markdown(f"""
<div class='result-box'>
    <span class='result-label'>Jenis Gonggong :</span>
    <span class='result-value'>{st.session_state.pred_class}</span>
</div>
<div class='result-box'>
    <span class='result-label'>Tingkat Akurasi :</span>
    <span class='result-value'>{st.session_state.conf_text}</span>
</div>
</div>""", unsafe_allow_html=True)
