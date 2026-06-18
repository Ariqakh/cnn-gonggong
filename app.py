import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import base64
from io import BytesIO
from rembg import remove
from tensorflow.keras.applications.mobilenet import preprocess_input

st.set_page_config(page_title="Klasifikasi Jenis Gonggong", layout="centered")

# --- CSS TETAP SAMA ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
[data-testid="stAppViewContainer"] { background: linear-gradient(180deg, #A7FFFF 0%, #D1FFFF 100%) !important; min-height: 100vh !important; }
.navbar { position: fixed; top: 0; left: 0; right: 0; z-index: 9999; background: #091a36; padding: 25px 20px; display: flex; align-items: center; gap: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
.navbar-title { color: #ffffff; font-size: 20px; font-weight: 600; }
.welcome-container { display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; min-height: 70vh; padding: 60px 20px 20px 20px; }
.welcome-title { font-size: 56px; font-weight: 800; color: #0b1d3a; margin-bottom: 15px; text-transform: uppercase; }
.cta-scroll-button { background: #0a3d3c; color: white !important; padding: 16px 36px; border-radius: 35px; font-weight: 700; text-decoration: none !important; }
.app-header { display: flex; align-items: center; justify-content: center; gap: 40px; margin: 60px 0 12px 0; }
.app-logo-img { width: 370px; height: 300px; }
.app-title-main { font-size: 65px; font-weight: 800; color: #0b1d3a; text-transform: uppercase; }
.img-preview-container { background: #E8E8E8; border-radius: 20px; margin: 15px auto; width: 100%; height: 260px; display: flex; align-items: center; justify-content: center; overflow: hidden; border: 1px solid #ddd; }
.img-preview-container img { max-width: 100%; max-height: 100%; object-fit: contain; }
div.stButton > button { color: white !important; font-weight: 700 !important; padding: 15px 25px !important; border-radius: 30px !important; border: none !important; background: #0a3d3c !important; }
.result-box { background-color: #87D4D4; border-radius: 20px; padding: 20px 25px; margin-top: 20px; }
.result-label { font-weight: 700; color: #0b1d3a; font-size: 18px; min-width: 180px; display: inline-block; }
.result-value { font-weight: 700; color: #0b1d3a; font-size: 18px; }
.warning-box { background-color: #FFDADA; color: #CC0000; padding: 10px; border-radius: 15px; margin-bottom: 10px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_my_model():
    from keras.models import load_model as keras_load_model
    # Daftarkan preprocess_input ke custom_objects agar sinkron
    custom_objs = {'preprocess_input': preprocess_input}
    return keras_load_model("model_gonggong.h5", custom_objects=custom_objs, compile=False)

model = load_my_model()
classes = ['Canarium Mutabile', 'Canarium Urseus', 'Laevistrombus Turturella', 'Pugilina Coclidium']

# --- TAMPILAN NAVBAR & HEADER ---
st.markdown("<div class='navbar'><span class='navbar-title'>Universitas Maritim Raja Ali Haji</span></div>", unsafe_allow_html=True)
st.markdown("<div class='welcome-container'><div class='welcome-title'>Sistem Klasifikasi Gonggong</div><a class='cta-scroll-button' href='#mulai-klasifikasi'>Klasifikasi Sekarang</a></div><div id='mulai-klasifikasi'></div>", unsafe_allow_html=True)

# --- CORE LOGIC ---
@st.fragment
def application_core():
    if "bg_removed_image" not in st.session_state: st.session_state.bg_removed_image = None
    if "pred_class" not in st.session_state: st.session_state.pred_class = "-"
    if "conf_text" not in st.session_state: st.session_state.conf_text = "-"
    if "warn_html" not in st.session_state: st.session_state.warn_html = ""

    uploaded_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is None:
        st.session_state.bg_removed_image = None
        st.session_state.pred_class = "-"
        st.session_state.conf_text = "-"
        st.session_state.warn_html = ""
    else:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Gambar Asli", use_column_width=True)
        with col2:
            if st.session_state.bg_removed_image:
                st.image(st.session_state.bg_removed_image, caption="Hasil Background Removed", use_column_width=True)
            else:
                st.info("Belum diproses")

        if st.session_state.bg_removed_image is None:
            if st.button("Hapus Latar Belakang"):
                with st.spinner("Proses..."):
                    output_img = remove(image)
                    bg_white = Image.new("RGB", output_img.size, (255, 255, 255))
                    bg_white.paste(output_img, mask=output_img.split()[3])
                    st.session_state.bg_removed_image = bg_white
                st.rerun()
        else:
            if st.button("Analisis Gambar"):
                with st.spinner("Menganalisis..."):
                    # 1. Sharpening
                    segmented_np = np.array(st.session_state.bg_removed_image)
                    img_bgr = cv2.cvtColor(segmented_np, cv2.COLOR_RGB2BGR)
                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                    sharpened = cv2.filter2D(img_bgr, -1, kernel)
                    final_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
                    
                    # 2. Resize & Preprocess (SINKRONISASI PENTING)
                    img_resized = Image.fromarray(final_rgb).resize((224, 224))
                    img_array = np.array(img_resized).astype("float32")
                    # Gunakan preprocess_input yang sama dengan model
                    img_input = preprocess_input(np.expand_dims(img_array, axis=0))
                    
                    # 3. Prediksi
                    prediction = model.predict(img_input)
                    max_conf = np.max(prediction)
                    
                    st.session_state.pred_class = classes[np.argmax(prediction)]
                    st.session_state.conf_text = f"{max_conf * 100:.2f} %"
                st.rerun()

    # Tampilkan Hasil
    st.markdown(f"""
    <div class='result-box'><span class='result-label'>Jenis Gonggong :</span><span class='result-value'>{st.session_state.pred_class}</span></div>
    <div class='result-box'><span class='result-label'>Tingkat Akurasi :</span><span class='result-value'>{st.session_state.conf_text}</span></div>
    """, unsafe_allow_html=True)

application_core()
