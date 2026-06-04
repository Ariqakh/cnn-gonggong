import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import base64
from io import BytesIO

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Klasifikasi Jenis Gonggong", layout="centered")

# --- CSS STYLING ---
st.markdown("""
<style>
/* CSS Anda tetap sama sesuai yang Anda lampirkan di app (5).py */
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_my_model():
    from keras.models import load_model as keras_load_model
    from keras.layers import Dense, InputLayer, Dropout

    # Custom handling untuk load model
    model = keras_load_model("model_gonggong.h5", compile=False)
    return model

model = load_my_model()
classes = ["Canarium Mutabile", "Canarium Urseus", "Laevistrombus Turturella", "Pugilina Coclidium"]

# --- FUNGSI PROSES GAMBAR ---
def process_bg(image):
    img_np = np.array(image)
    gray_np = 0.2989 * img_np[:,:,0] + 0.5870 * img_np[:,:,1] + 0.1140 * img_np[:,:,2]
    background_mask = gray_np < 40 
    segmented_np = img_np.copy()
    segmented_np[background_mask] = [255, 255, 255]
    return Image.fromarray(segmented_np)

# --- TAMPILAN APP ---
# (Navbar, Header, dll sesuai file Anda...)

uploaded_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# Session State
if "is_bg_removed" not in st.session_state: st.session_state.is_bg_removed = False

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    processed_image = process_bg(image)
    
    # Menampilkan Gambar
    col1, col2 = st.columns(2)
    with col1: st.image(image, caption="Gambar Asli")
    with col2: 
        if st.session_state.is_bg_removed: st.image(processed_image, caption="Hasil Hapus Background")
        else: st.write("Belum Diproses")

    # Tombol
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Hapus Latar Belakang"):
            st.session_state.is_bg_removed = True
            st.rerun()
    with col_btn2:
        analyze_clicked = st.button("Analisis Gambar")

    if analyze_clicked:
        final_img = processed_image if st.session_state.is_bg_removed else image
        img_resized = final_img.resize((224, 224))
        img_array = np.array(img_resized).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        max_conf = np.max(prediction)
        
        # Validasi
        if max_conf < 0.50:
            st.warning("⚠️ Gambar tidak dikenali sebagai Gonggong.")
        else:
            pred_class = classes[np.argmax(prediction)]
            st.success(f"Jenis: {pred_class} ({max_conf*100:.2f}%)")
