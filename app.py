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

/* Hilangkan padding bawah bawaan streamlit block yang bikin scroll kosong kepanjangan */
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

/* BASE STYLING FOR FILE UPLOADER (DESKTOP) */
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

/* STRUKTUR HASIL PREDIKSI (DESKTOP) */
.result-box {
    background-color: #87D4D4;
    border-radius: 20px;
    padding: 20px 25px;
    margin-top: 30px;
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

/* PERBAIKAN POSISI FOOTER */
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

    /* KUSTOMISASI TOMBOL UPLOAD DI MOBILE SESUAI GAMBAR REFERENSI */
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
    /* Sembunyikan ikon seret bawaan drag&drop */
    [data-testid="stFileUploader"] section svg {
        display: none !important;
    }
    /* Mengubah tombol internal Streamlit menjadi style minimalis putih */
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
    /* Sembunyikan pesan teks seret bawaan browser */
    [data-testid="stFileUploader"] section > input + div {
        display: none !important;
    }
    /* Tampilkan label kustom di sebelah kanan tombol */
    [data-testid="stFileUploader"] section::after {
        content: "200MB per file • JPG, PNG";
        font-size: 14px;
        color: #777777;
        font-weight: 400;
        display: inline-block;
    }

    div.stButton > button {
        width: 100% !important;
        font-size: 18px !important;
        padding: 15px 20px !important;
    }
    
    /* MENYELARASKAN TANDA HUBUNG / TITIK DUA HASIL PREDIKSI DI HP */
    .result-box {
        padding: 15px 20px;
        margin-top: 15px;
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
        margin-top: 25px; 
    }
    .white-footer-canvas { 
        margin-top: 50px !important;
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

    # PATCH Dense
    original_dense = Dense.from_config
    @classmethod
    def custom_dense(cls, config):
        config.pop("quantization_config", None)
        return original_dense(config)
    Dense.from_config = custom_dense

    # PATCH InputLayer
    original_input = InputLayer.from_config
    @classmethod
    def custom_input(cls, config):
        config.pop("batch_shape", None)
        config.pop("optional", None)
        if "batch_input_shape" not in config:
            config["batch_input_shape"] = [None, 224, 224, 3]
        return cls(**config)
    InputLayer.from_config = custom_input

    # PATCH Dropout
    original_dropout = Dropout.from_config
    @classmethod
    def custom_dropout(cls, config):
        config.pop("seed_generator", None)
        return original_dropout(config)
    Dropout.from_config = custom_dropout

    # LOAD MODEL
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
    <span class="navbar-title">Universitas Maritim Raja Ali Haji</span>
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

# --- MANAGING SESSION STATE FOR RESET OUTCOMES ---
if "pred_class" not in st.session_state:
    st.session_state.pred_class = "-"
if "conf_text" not in st.session_state:
    st.session_state.conf_text = "-"
if "warn_box_html" not in st.session_state:
    st.session_state.warn_box_html = ""

# --- FILE UPLOADER COMPONENT ---
uploaded_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# ==========================================================================
# PERBAIKAN TOTAL ELEMEN FILE UPLOADER (MENAMPILKAN SILANG & FIX RESET HP)
# ==========================================================================
if uploaded_file is None:
    st.session_state.pred_class = "-"
    st.session_state.conf_text = "-"
    st.session_state.warn_box_html = ""
else:
    # JIKA FILE SUDAH BERHASIL DI-UPLOAD
    st.markdown("""
    <style>
    /* Hilangkan teks pembatas ukuran file default */
    [data-testid="stFileUploader"] section::after { 
        content: "" !important; 
        display: none !important; 
    }

    @media (max-width: 480px) {
        /* Sembunyikan tombol 'Browse files' ketika file sudah masuk */
        [data-testid="stFileUploader"] section button {
            display: none !important;
        }

        /* Tata letak pembungkus nama file dipaksa rata kiri penuh */
        [data-testid="stFileUploader"] section {
            display: flex !important;
            flex-direction: row !important;
            justify-content: space-between !important;
            align-items: center !important;
            padding: 12px 20px !important;
            background-color: #EAEAEA !important;
        }

        /* Sembunyikan elemen bawaan ikon berkas, svg, dan tanda tambah (+) liar */
        [data-testid="stFileUploader"] section data,
        [data-testid="stFileUploader"] section svg,
        [data-testid="stFileUploader"] section span {
            display: none !important;
        }

        /* Paksa container penampung nama file asli muncul rapi di sisi kiri */
        [data-testid="stFileUploader"] section > input + div {
            display: flex !important;
            flex-direction: row !important;
            align-items: center !important;
            justify-content: space-between !important;
            width: 100% !important;
            margin: 0 !important;
        }

        /* PAKSA TOMBOL SILANG BAWAAN (X) MENJADI AKTIF, SANGAT JELAS, DAN BISA DIKLIK */
        [data-testid="stFileUploader"] button[aria-label="Remove file"],
        [data-testid="stFileUploader"] button[data-testid="stFileUploaderDeleteBtn"] {
            display: inline-block !important;
            visibility: visible !important;
            opacity: 1 !important;
            position: relative !important;
            background-color: #DCDCDC !important; /* Latar belakang tombol silang abu lingkaran */
            color: #333333 !important;
            border-radius: 50% !important;
            width: 28px !important;
            height: 28px !important;
            border: none !important;
            margin: 0 0 0 auto !important; /* Geser mentok kanan sesuai foto */
            padding: 0 !important;
            line-height: 26px !important;
            text-align: center !important;
            box-shadow: 0px 1px 2px rgba(0,0,0,0.15) !important;
            z-index: 999 !important;
        }

        /* Cetak karakter huruf X besar di dalam tombol lingkaran tersebut */
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

# --- IMAGE PREVIEW CONTROLLER ---
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

# --- KONTROL LOGIKA DAN VALIDASI GAMBAR PERKETAT ---
if analyze_clicked:
    if uploaded_file is not None:
        # 1. FITUR HAPUS BACKGROUND (SEGMENTASI INTENSITAS CITRA)
        # Mengubah citra masukan menjadi format matriks NumPy
        img_np = np.array(image)
        
        # Ekstraksi komponen nilai keabuan (Grayscale secara manual demi efisiensi)
        gray_np = 0.2989 * img_np[:,:,0] + 0.5870 * img_np[:,:,1] + 0.1140 * img_np[:,:,2]
        
        # Membuat masking threshold: piksel latar belakang terang (> 200) diisolasi
        background_mask = gray_np > 200
        
        # Mengkloning gambar asli dan mengubah area terisolasi menjadi warna hitam murni [0, 0, 0]
        segmented_np = img_np.copy()
        segmented_np[background_mask] = [0, 0, 0]
        
        # Mengembalikan matriks prapemrosesan ke bentuk PIL Image objek
        processed_image = Image.fromarray(segmented_np)

        # 2. INPUT GAMBAR HASIL PEMPROSESAN KE MODEL CNN MOBILENET
        img_resized = processed_image.resize((224, 224))
        img_array = np.array(img_resized).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        max_conf = np.max(prediction)
        
        # MEMPERKETAT DETEKSI: Naikkan ambang batas ke 0.65 (65%) + Filter Gambar Kosong Sembarang
        pure_white = np.sum(np.all(img_np >= 245, axis=-1))
        total_pixels = img_np.shape[0] * img_np.shape[1]
        
        if max_conf < 0.65 or (pure_white / total_pixels) > 0.40:
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

# Render pesan peringatan jika terdeteksi non-gonggong / akurasi rendah
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
