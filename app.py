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

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    margin: 0;
    padding: 0;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #A7FFFF 0%, #D1FFFF 100%) !important;
    min-height: 100vh !important;
}

[data-testid="stMain"] { background: transparent !important; }

header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { 
    visibility: hidden; display: none !important; 
}

.navbar {
    position: fixed; top: 0; left: 0; right: 0; z-index: 9999;
    background: #091a36; padding: 25px 20px;
    display: flex; align-items: center; gap: 12px;
}
.navbar-title { color: #ffffff; font-size: 20px; font-weight: 600; }

.app-header {
    display: flex; align-items: center; justify-content: center;
    gap: 20px; margin-top: 80px; margin-bottom: 20px;
}
.app-logo-img { width: 150px; }
.app-title-main { font-size: 40px; font-weight: 800; color: #0b1d3a; line-height: 1; text-transform: uppercase; }

.img-preview-container {
    background: #E8E8E8; border-radius: 20px; margin: 10px auto;
    width: 100%; height: 200px; display: flex; align-items: center; justify-content: center;
    overflow: hidden; border: 1px solid #ddd;
}
.img-preview-container img { max-width: 100%; max-height: 100%; object-fit: contain; }

/* Perbaikan Button Styling */
div.stButton > button {
    background-color: #2D6A6A !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 30px !important;
    border: none !important;
    width: 100% !important;
    padding: 12px !important;
}
button[key="btn_remove_bg"] { background-color: #4A90E2 !important; }

.result-box {
    background-color: #87D4D4; border-radius: 20px;
    padding: 15px; margin-top: 10px; display: flex; justify-content: space-between;
}

/* MOBILE FIX */
@media (max-width: 480px) {
    .app-header { margin-top: 60px !important; flex-direction: column; gap: 10px; }
    .page-wrapper { margin-top: -20px !important; } /* Naikkan elemen ke atas */
    .button-group { gap: 10px !important; }
}
</style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_my_model():
    from keras.models import load_model
    return load_model("model_gonggong.h5", compile=False)

model = load_my_model()
classes = ["Canarium Mutabile", "Canarium Urseus", "Laevistrombus Turturella", "Pugilina Coclidium"]

# Session State
if "is_bg_removed" not in st.session_state: st.session_state.is_bg_removed = False
if "pred_class" not in st.session_state: st.session_state.pred_class = "-"
if "conf_text" not in st.session_state: st.session_state.conf_text = "-"

st.markdown("<div class='page-wrapper'>", unsafe_allow_html=True)

# UI
uploaded_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Logika Hapus Background
    img_np = np.array(image)
    gray_np = 0.2989 * img_np[:,:,0] + 0.5870 * img_np[:,:,1] + 0.1140 * img_np[:,:,2]
    background_mask = gray_np < 40 
    segmented_np = img_np.copy()
    segmented_np[background_mask] = [255, 255, 255]
    processed_image = Image.fromarray(segmented_np)

    col1, col2 = st.columns(2)
    with col1: st.image(image, caption="Asli", use_container_width=True)
    with col2:
        if st.session_state.is_bg_removed:
            st.image(processed_image, caption="Tanpa BG", use_container_width=True)
        else:
            st.info("Klik tombol di bawah")

    # Tombol Kontrol
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Hapus Latar Belakang", key="btn_remove_bg"):
            st.session_state.is_bg_removed = True
            st.rerun()

    # Tombol Analisis hanya muncul jika sudah hapus BG
    if st.session_state.is_bg_removed:
        with col_btn2:
            if st.button("Analisis Gambar", key="btn_analyze"):
                img_resized = processed_image.resize((224, 224))
                img_array = np.expand_dims(np.array(img_resized).astype("float32") / 255.0, axis=0)
                prediction = model.predict(img_array)
                predicted_index = np.argmax(prediction)
                st.session_state.pred_class = classes[predicted_index]
                st.session_state.conf_text = f"{np.max(prediction) * 100:.2f}%"

    st.markdown(f"""
    <div class='result-box'>
        <b>Jenis:</b> {st.session_state.pred_class} | <b>Akurasi:</b> {st.session_state.conf_text}
    </div>
    """, unsafe_allow_html=True)

else:
    st.session_state.is_bg_removed = False
    st.session_state.pred_class = "-"
    st.session_state.conf_text = "-"

st.markdown("</div>", unsafe_allow_html=True)
