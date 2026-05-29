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
    min-height: 100vh !important;
}

[data-testid="stMain"] {
    background: transparent !important;
    padding-bottom: 20px !important; 
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

/* ===== BUTTON STYLING ===== */
.button-group {
    display: flex;
    flex-direction: row;
    justify-content: center;
    gap: 15px;
    margin: 20px auto;
    width: 100%;
}

/* Default semua tombol */
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

/* Tombol Hapus Latar Belakang diubah ke hijau tua menyesuaikan logo */
div[data-testid="stButton"]:has(button[kind="secondary"]) > button,
div.stButton > button[kind="secondary"] {
    background: linear-gradient(135deg, #0a3d3c 0%, #166d62 100%) !important;
}

/* Fallback — tombol pertama hijau tua, tombol kedua hijau tua sedikit lebih terang */
div.stButton:nth-of-type(1) > button {
    background: linear-gradient(135deg, #0a3d3c 0%, #166d62 100%) !important;
}

div.stButton > button#btn_remove_bg,
div.stButton > button[key="btn_remove_bg"] {
    background: linear-gradient(135deg, #0a3d3c 0%, #166d62 100%) !important;
}

div.stButton > button#btn_analyze,
div.stButton > button[key="btn_analyze"] {
    background: linear-gradient(135deg, #115c5a 0%, #1a8375 100%) !important;
}

div.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(0,0,0,0.22) !important;
}

div.stButton > button:active {
    transform: translateY(0px) !important;
    opacity: 1 !important;
}

/* ===== HASIL PREDIKSI ===== */
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

.page-wrapper {
    margin-bottom: 0px;
    padding-bottom: 0px;
}

.white-footer-canvas {
    position: relative !important;
    margin-top: 80px !important;
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

/* ===== RESPONSIVE MOBILE ===== */
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
        gap: 10px;
        text-align: center;
        margin-top: 30px;
        margin-bottom: 6px;
    }
    .app-logo-img { 
        width: 150px; 
        height: auto; 
    }
    .app-title-main { 
        font-size: 32px; 
    }
    .app-subtitle-main {
        font-size: 12px;
    }
    .img-preview-container { 
        height: 160px; 
        margin: 8px auto;
    }
    
    .button-group {
        flex-direction: column !important;
        gap: 10px;
        margin: 10px auto !important;
    }

    [data-testid="stFileUploader"] section {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 15px !important;
        padding: 12px 20px !important;
        background-color: #EAEAEA !important;
        border: none !important;
        border-radius: 40px !important;
    }
    [data-testid="stFileUploader"] section svg {
        display: none !important;
    }
    [data-testid="stFileUploader"] section button {
        background-color: #FFFFFF !important;
        color: #333333 !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 12px !important;
        padding: 6px 16px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        margin: 0 !important;
        box-shadow: 0px 1px 3px rgba(0,0,0,0.1) !important;
    }
    [data-testid="stFileUploader"] section > input + div {
        display: none !important;
    }
    [data-testid="stFileUploader"] section::after {
        content: "200MB per file • JPG, PNG";
        font-size: 14px;
        color: #777777;
        font-weight: 400;
        display: inline-block;
    }

    div.stButton > button {
        width: 100% !important;
        font-size: 15px !important;
        padding: 12px 20px !important;
    }
    
    .result-box {
        padding: 15px 20px;
        margin-top: 12px;
        display: flex !important;
        flex-direction: row !important;
        justify-content: flex-start !important;
        align-items: center !important;
    }
    .result-label {
        font-size: 15px;
        min-width: 135px; 
        margin-right: 0px;
    }
    .result-value {
        font-size: 15px;
    }
    .page-wrapper { 
        margin-top: 15px; 
    }
    .white-footer-canvas { 
        margin-top: 40px !important;
        padding: 15px 0px !important;
    }
    .footer-text {
        font-size: 11px;
    }
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

    model = keras_load_model("model_gonggong.h5", compile=False)
    return model

model = load_my_model()

classes = [
    "Canarium Mutabile",
    "Canarium Urseus",
    "Laevistrombus Turturella",
    "Pugilina Coclidium"
]

try:
    with open("logo_umrah.png", "rb") as f:
        nav_logo_bytes = f.read()
    encoded_nav_logo = base64.b64encode(nav_logo_bytes).decode()
except:
    encoded_nav_logo = ""

st.markdown(f"""
<div class="navbar">
    <div class="navbar-logo">
        <img src="data:image/png;base64,{encoded_nav_logo}" width="32" height="32" style="object-fit:cover;">
    </div>
    <span class="navbar-title">Universitas Maritime Raja Ali Haji</span>
</div>
<div class="navbar-spacer"></div>
""", unsafe_allow_html=True)

st.markdown("<div class='page-wrapper'>", unsafe_allow_html=True)

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

# --- SESSION STATE ---
if "pred_class" not in st.session_state:
    st.session_state.pred_class = "-"
if "conf_text" not in st.session_state:
    st.session_state.conf_text = "-"
if "warn_box_html" not in st.session_state:
    st.session_state.warn_box_html = ""
if "is_bg_removed" not in st.session_state:
    st.session_state.is_bg_removed = False

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is None:
    st.session_state.pred_class = "-"
    st.session_state.conf_text = "-"
    st.session_state.warn_box_html = ""
    st.session_state.is_bg_removed = False
else:
    st.markdown("""
    <style>
    [data-testid="stFileUploader"] section::after { 
        content: "" !important; 
        display: none !important; 
    }

    @media (max-width: 480px) {
        [data-testid="stFileUploader"] section button {
            display: none !important;
        }

        [data-testid="stFileUploader"] section {
            display: flex !important;
            flex-direction: row !important;
            justify-content: space-between !important;
            align-items: center !important;
            padding: 12px 20px !important;
            background-color: #EAEAEA !important;
        }

        [data-testid="stFileUploader"] section data,
        [data-testid="stFileUploader"] section svg,
        [data-testid="stFileUploader"] section span {
            display: none !important;
        }

        [data-testid="stFileUploader"] section > input + div {
            display: flex !important;
            flex-direction: row !important;
            align-items: center !important;
            justify-content: space-between !important;
            width: 100% !important;
            margin: 0 !important;
        }

        [data-testid="stFileUploader"] button[aria-label="Remove file"],
        [data-testid="stFileUploader"] button[data-testid="stFileUploaderDeleteBtn"] {
            display: inline-block !important;
            visibility: visible !important;
            opacity: 1 !important;
            position: relative !important;
            background-color: #DCDCDC !important;
            color: #333333 !important;
            border-radius: 50% !important;
            width: 28px !important;
            height: 28px !important;
            border: none !important;
            margin: 0 0 0 auto !important;
            padding: 0 !important;
            line-height: 26px !important;
            text-align: center !important;
            box-shadow: 0px 1px 2px rgba(0,0,0,0.15) !important;
            z-index: 999 !important;
        }

        [data-testid="stFileUploader"] button[aria-label="Remove file"]::after {
            content: "✕" !important;
            font-size: 14px !important;
            font-weight: 800 !important;
            color: #222222 !important;
            display: block !important;
            text-align: center !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# --- IMAGE PREVIEW ---
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
        buffered1 = BytesIO()
        image.save(buffered1, format="JPEG")
        img_str1 = base64.b64encode(buffered1.getvalue()).decode()
        st.markdown(f"""
        <div class='img-preview-container'>
            <img src="data:image/jpeg;base64,{img_str1}">
        </div>
        <div style='text-align:center; font-weight:600; color:#0b1d3a; font-size:13px;'>Gambar Asli</div>
        """, unsafe_allow_html=True)
        
    with col2:
        if st.session_state.is_bg_removed:
            buffered2 = BytesIO()
            processed_image.save(buffered2, format="JPEG")
            img_str2 = base64.b64encode(buffered2.getvalue()).decode()
            img_html = f'<img src="data:image/jpeg;base64,{img_str2}">'
        else:
            img_html = "<span class='img-placeholder-text'>Belum Diproses</span>"
            
        st.markdown(f"""
        <div class='img-preview-container'>
            {img_html}
        </div>
        <div style='text-align:center; font-weight:600; color:#0b1d3a; font-size:13px;'>Hasil Hapus Background</div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class='img-preview-container'>
        <span class='img-placeholder-text'>Gambar</span>
    </div>
    """, unsafe_allow_html=True)

# --- TOMBOL ---
st.markdown("<div class='button-group'>", unsafe_allow_html=True)

if not st.session_state.is_bg_removed:
    bg_clicked = st.button("Hapus Latar Belakang", key="btn_remove_bg")
    analyze_clicked = False
else:
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        bg_clicked = st.button("Hapus Latar Belakang", key="btn_remove_bg")
    with col_btn2:
        analyze_clicked = st.button("Analisis Gambar", key="btn_analyze")

st.markdown("</div>", unsafe_allow_html=True)

if bg_clicked:
    if uploaded_file is not None:
        st.session_state.is_bg_removed = True
        st.rerun()
    else:
        st.warning("Silakan upload gambar terlebih dahulu.")

if analyze_clicked:
    if uploaded_file is not None:
        img_np = np.array(image)
        gray_np = 0.2989 * img_np[:,:,0] + 0.5870 * img_np[:,:,1] + 0.1140 * img_np[:,:,2]
        background_mask = gray_np < 40
        segmented_np = img_np.copy()
        segmented_np[background_mask] = [255, 255, 255]
        final_processed = Image.fromarray(segmented_np)

        img_resized = final_processed.resize((224, 224))
        img_array = np.array(img_resized).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        max_conf = np.max(prediction)
        
        pure_white = np.sum(np.all(segmented_np >= 245, axis=-1))
        total_pixels = segmented_np.shape[0] * segmented_np.shape[1]
        
        if max_conf < 0.50 or (pure_white / total_pixels) > 0.85:
            st.session_state.warn_box_html = "<div class='warning-box'>⚠️ Gambar tidak dikenali sebagai Gonggong. Harap upload foto Gonggong yang jelas.</div>"
            st.session_state.pred_class = "-"
            st.session_state.conf_text = "-"
        else:
            st.session_state.warn_box_html = ""
            predicted_index = np.argmax(prediction)
            st.session_state.pred_class = classes[predicted_index]
            st.session_state.conf_text = f"{max_conf * 100:.2f}%"
    else:
        st.warning("Silakan upload gambar terlebih dahulu.")

if st.session_state.warn_box_html:
    st.markdown(st.session_state.warn_box_html, unsafe_allow_html=True)

st.markdown(f"""
<div class='result-box'>
    <span class='result-label'>Jenis Gonggong :</span>
    <span class='result-value'>{st.session_state.pred_class}</span>
</div>
<div class='result-box'>
    <span class='result-label'>Tingkat Akurasi :</span>
    <span class='result-value'>{st.session_state.conf_text}</span>
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
