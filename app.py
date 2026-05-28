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
    height: 100%;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #A7FFFF 0%, #D1FFFF 100%) !important;
    min-height: 100vh;
    position: relative;
}

[data-testid="stMain"] {
    background: transparent !important;
    padding-bottom: 120px !important; /* Memberikan ruang agar tidak tertimpa footer */
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
    margin: 35px auto;
    width: 100%;
    height: 300px; 
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
    font-size: 18px;
    color: #666;
    font-weight: 500;
}

/* KUSTOMISASI KOTAK UPLOAD FILE */
[data-testid="stFileUploader"] section {
    background-color: #F3F3F3 !important;
    border: 1px solid #ccc !important;
    border-radius: 30px !important;
    padding: 25px 15px !important;
}
[data-testid="stFileUploader"] section > input + div {
    display: none !important; /* Sembunyikan teks default Streamlit */
}
[data-testid="stFileUploader"] section::after {
    content: "Upload file JPG, PNG";
    font-size: 16px;
    font-weight: 600;
    color: #555555;
    display: block;
    text-align: center;
    cursor: pointer;
}

div.stButton > button {
    background-color: #2D6A6A !important;
    color: white !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    padding: 20px 40px !important;
    border-radius: 30px !important;
    border: none !important;
    margin: 10px auto !important;
    display: block !important;
    width: 200px;
}

/* PERBAIKAN TAMPILAN HASIL PREDIKSI AGAR RAPI SEJAJAR */
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
    min-width: 160px; /* Mengunci lebar teks label agar tanda hubung/titik dua sejajar */
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

.page-wrapper {
    margin-bottom: 0px;
    padding-bottom: 0px;
}

/* FOOTER MENTOK PALING BAWAH HALAMAN */
.white-footer-canvas {
    position: absolute;
    bottom: 0; 
    left: 0; 
    right: 0;
    padding: 30px 0px;
    background: transparent;
    display: flex;
    justify-content: center; 
    align-items: center;
    z-index: 10;
}

.footer-text {
    text-align: center;
    color: #1a364a;
    font-size: 14px;
    font-weight: 500;
    line-height: 1.5;
    margin: 0 auto;
}

/* RESPONSIVE MOBILE OPTIMIZATION */
@media (max-width: 480px) {
    .navbar {
        padding: 15px 15px;
        gap: 8px;
    }
    .navbar-title {
        font-size: 14px;
    }
    .app-header {
        flex-direction: column;
        gap: 15px;
        text-align: center;
        margin-top: 40px;
    }
    .app-logo-img { 
        width: 180px; 
        height: auto; 
    }
    .app-title-main { 
        font-size: 32px; 
    }
    .app-subtitle-main {
        font-size: 12px;
    }
    .img-preview-container { 
        height: 220px; 
        margin: 20px auto;
    }
    [data-testid="stFileUploader"] section {
        padding: 20px 10px !important;
    }
    [data-testid="stFileUploader"] section::after {
        font-size: 14px;
    }
    div.stButton > button {
        width: 100% !important;
        font-size: 18px !important;
        padding: 15px 20px !important;
    }
    .result-box {
        padding: 15px;
        margin-top: 15px;
        flex-direction: row; /* Tetap horizontal di HP agar sejajar */
        justify-content: flex-start;
    }
    .result-label {
        font-size: 15px;
        min-width: 130px; /* Lebar label disesuaikan untuk layar HP */
    }
    .result-value {
        font-size: 15px;
    }
    .page-wrapper { 
        margin-top: 25px; 
    }
    .white-footer-canvas { 
        padding: 20px 0px;
    }
    .footer-text {
        font-size: 11px;
    }
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_my_model():

    import tensorflow as tf
    from keras.models import load_model as keras_load_model
    from keras.layers import Dense, InputLayer, Dropout

    # =========================
    # PATCH Dense
    # =========================
    original_dense = Dense.from_config

    @classmethod
    def custom_dense(cls, config):

        config.pop("quantization_config", None)

        return original_dense(config)

    Dense.from_config = custom_dense

    # =========================
    # PATCH InputLayer
    # =========================
    original_input = InputLayer.from_config

    @classmethod
    def custom_input(cls, config):

        config.pop("batch_shape", None)
        config.pop("optional", None)

        if "batch_input_shape" not in config:
            config["batch_input_shape"] = [None, 224, 224, 3]

        return cls(**config)

    InputLayer.from_config = custom_input

    # =========================
    # PATCH Dropout
    # =========================
    original_dropout = Dropout.from_config

    @classmethod
    def custom_dropout(cls, config):

        config.pop("seed_generator", None)

        return original_dropout(config)

    Dropout.from_config = custom_dropout

    # =========================
    # LOAD MODEL
    # =========================
    model = keras_load_model(
        "model_gonggong.h5",
        compile=False
    )

    return model

model = load_my_model()

classes = [
    "Canarium Mutabile",
    "Canarium Urseus",
    "Laevistrombus Turturella",
    "Pugilina Coclidium"
]

with open("logo_umrah.png", "rb") as f:
    nav_logo_bytes = f.read()
encoded_nav_logo = base64.b64encode(nav_logo_bytes).decode()

st.markdown(f"""
<div class="navbar">
    <div class="navbar-logo">
        <img src="data:image/png;base64,{encoded_nav_logo}" width="32" height="32" style="object-fit:cover;">
    </div>
    <span class="navbar-title">Universitas Maritim Raja Ali Haji</span>
</div>
<div class="navbar-spacer"></div>
""", unsafe_allow_html=True)

st.markdown("<div class='page-wrapper'>", unsafe_allow_html=True)

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

uploaded_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    st.markdown(f"""
    <div class='img-preview-container'>
        <img src="data:image/jpeg;base64,{img_str}">
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class='img-preview-container'>
        <span class='img-placeholder-text'>Gambar</span>
    </div>
    """, unsafe_allow_html=True)

analyze_clicked = st.button("Analisis Gambar")

predicted_class = "-"
confidence_text = "-"
is_gonggong = True

if analyze_clicked:
    if uploaded_file is not None:
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        max_conf = np.max(prediction)
        
        if max_conf < 0.60:
            is_gonggong = False
            st.markdown("<div class='warning-box'>⚠️ Gambar tidak dikenali sebagai Gonggong. Harap upload foto Gonggong yang jelas.</div>", unsafe_allow_html=True)
        else:
            predicted_index = np.argmax(prediction)
            predicted_class = classes[predicted_index]
            confidence_text = f"{max_conf * 100:.2f}%"
    else:
        st.warning("Silakan upload gambar terlebih dahulu.")

st.markdown(f"""
<div class='result-box'>
    <span class='result-label'>Jenis Gonggong :</span>
    <span class='result-value'>{predicted_class}</span>
</div>
<div class='result-box'>
    <span class='result-label'>Tingkat Akurasi :</span>
    <span class='result-value'>{confidence_text}</span>
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True) 

st.markdown("""
<div class='white-footer-canvas'>
    <div class='footer-text'>
        © 2026 Sistem Klasifikasi Jenis Gonggong berbasis CNN MobileNet<br>
        Fakultas Teknik dan Teknologi Kemaritiman - UMRAH
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
