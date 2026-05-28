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
}

[data-testid="stMain"] {
    background: transparent !important;
    height: 100%;
}

/* Mengatur container utama agar footer mentok di bawah menggunakan flexbox */
[data-testid="stMain"] > div {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
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

.page-wrapper {
    flex: 1;
    display: flex;
    flex-direction: column;
    margin-top: 100px;
    padding-bottom: 40px;
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

[data-testid="stFileUploader"] section {
    background-color: #F3F3F3 !important;
    border: 1px solid #ccc !important;
    border-radius: 30px !important;
    padding: 15px !important;
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

/* Desain baru tampilan hasil prediksi yang rapi */
.result-container {
    background-color: #87D4D4;
    border-radius: 20px;
    padding: 25px;
    margin-top: 30px;
    display: grid;
    grid-template-columns: auto 1fr;
    row-gap: 15px;
    column-gap: 15px;
    align-items: center;
}
.result-label {
    font-weight: 700;
    color: #0b1d3a;
    font-size: 18px;
}
.result-value {
    font-weight: 700;
    color: #0b1d3a;
    font-size: 18px;
}

.warning-box {
    background-color: #FFDADA;
    color: #CC0000;
    padding: 15px;
    border-radius: 15px;
    font-size: 14px;
    margin-bottom: 15px;
    font-weight: 600;
}

/* Footer dipaksa mentok ke batas paling bawah halaman */
.white-footer-canvas {
    margin-top: auto;
    width: 100%;
    padding: 40px 20px;
    display: flex;
    justify-content: center; 
    align-items: center;
    background: transparent;
}

.footer-text {
    text-align: center;
    color: #1a364a;
    font-size: 14px;
    font-weight: 500;
    line-height: 1.5;
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
    .page-wrapper {
        margin-top: 75px;
    }
    .app-header {
        flex-direction: column;
        gap: 15px;
        text-align: center;
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
    div.stButton > button {
        width: 100% !important;
        font-size: 18px !important;
        padding: 15px 20px !important;
    }
    .result-container {
        grid-template-columns: 1fr;
        row-gap: 8px;
        padding: 20px;
        margin-top: 20px;
    }
    .result-label {
        font-size: 15px;
        margin-bottom: -4px;
    }
    .result-value {
        font-size: 17px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(11, 29, 58, 0.15);
    }
    .result-value:last-child {
        border-bottom: none;
        padding-bottom: 0;
    }
    .white-footer-canvas { 
        padding: 30px 15px;
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
<div class='result-container'>
    <div class='result-label'>Jenis Gonggong :</div>
    <div class='result-value'>{predicted_class}</div>
    <div class='result-label'>Tingkat Akurasi :</div>
    <div class='result-value'>{confidence_text}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True) 
st.markdown("</div>", unsafe_allow_html=True) 

st.markdown("""
<div class='white-footer-canvas'>
    <div class='footer-text'>
        © 2026 Sistem Klasifikasi Jenis Gonggong berbasis CNN MobileNet<br>
        Fakultas Teknik dan Teknologi Kemaritiman - UMRAH
    </div>
</div>
""", unsafe_allow_html=True)
