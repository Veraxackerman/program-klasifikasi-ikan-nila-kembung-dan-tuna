import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

st.set_page_config(
    page_title="Sistem Klasifikasi Kesegaran Ikan",
    layout="centered"
)

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

body {
    background-color: #111827;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.block-container {
    padding-top: 2rem;
    max-width: 900px;
}

.app-header {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
    color: white;
}

.app-header h1 {
    font-size: 2rem;
    margin-bottom: .4rem;
    font-weight: 800;
}

.app-header p {
    color: rgba(255,255,255,.8);
    margin: 0;
}

[data-testid="stImage"] img {
    border-radius: 14px;
}

[data-testid="stFileUploader"] {
    border: 2px dashed #475569;
    border-radius: 16px;
    padding: 8px;
    background: #1e293b;
}

@media (max-width: 768px) {

    .app-header {
        padding: 1.5rem;
    }

    .app-header h1 {
        font-size: 1.6rem;
    }

    .app-header p {
        font-size: 12px;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

}

</style>
""", unsafe_allow_html=True)

IMG_SIZE = (224, 224)
THRESHOLD = 0.5

# Kata kunci IKAN
FISH_ALLOW = {
    'fish', 'tench', 'goldfish', 'salmon', 'eel', 'shark', 'ray', 'stingray',
    'puffer', 'blowfish', 'lionfish', 'coho', 'carp', 'mackerel', 'tuna',
    'cod', 'trout', 'snapper', 'grouper', 'flounder', 'sole', 'halibut',
    'pike', 'barracuda', 'marlin', 'swordfish', 'anchovy', 'herring',
    'sardine', 'catfish', 'tilapia',
}

# Kata kunci BUKAN IKAN
NON_FISH_BLOCK = {

    # Manusia & atribut
    'person', 'people', 'man', 'woman', 'boy', 'girl', 'face', 'head',
    'human', 'baby', 'child', 'portrait',

    # Pakaian
    'shirt', 'dress', 'suit', 'jacket', 'coat', 'trousers', 'jeans',
    'skirt', 'blouse', 'uniform', 'jersey', 'robe', 'kimono', 'abaya',
    'hijab', 'veil', 'scarf', 'tie', 'sock', 'shoe', 'boot', 'sandal',
    'hat', 'cap', 'helmet', 'glasses', 'sunglasses', 'bag', 'backpack',
    'handbag', 'purse', 'wallet', 'umbrella', 'watch',

    # Hewan
    'cat', 'dog', 'horse', 'cow', 'elephant', 'lion', 'tiger', 'bear',
    'rabbit', 'bird', 'chicken', 'duck', 'goose', 'penguin', 'parrot',
    'snake', 'lizard', 'frog', 'insect', 'butterfly', 'bee', 'monkey',

    # Kendaraan
    'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train', 'airplane',

    # Tempat
    'building', 'house', 'bridge', 'street', 'road',
    'mountain', 'forest', 'tree', 'flower',

    # Elektronik
    'phone', 'computer', 'keyboard', 'television',
    'camera', 'monitor', 'laptop', 'screen',

    # Benda
    'book', 'bottle', 'cup', 'chair', 'table',
    'lamp', 'clock', 'sofa',

    # Media / gambar
    'comic', 'cartoon', 'illustration', 'drawing',
    'anime', 'poster', 'banner',

    # Musik
    'piano', 'guitar', 'violin', 'drum', 'stage',

    # Diagram
    'web', 'diagram', 'chart', 'graph', 'plot',

    # Seafood non ikan
    'shrimp', 'lobster', 'crab',
    'squid', 'octopus', 'seafood',
}

IGNORE_LABELS = {
    'plate', 'tray', 'platter', 'dish', 'bowl', 'strainer', 'colander',
    'wok', 'frying pan', 'hotpot', 'cutting board', 'paper towel',
    'menu', 'table', 'napkin', 'server',
}

@st.cache_resource
def load_classifier():

    model_paths = [
        "model_final.keras",
        "output/model_final.keras",
        "model_final.h5"
    ]

    for path in model_paths:

        if os.path.exists(path):

            model = tf.keras.models.load_model(
                path,
                compile=False
            )

            return model, path

    return None, None

@st.cache_resource
def load_validator():

    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import (
        preprocess_input,
        decode_predictions
    )

    validator = MobileNetV2(
        weights="imagenet",
        include_top=True
    )

    return validator, preprocess_input, decode_predictions

def validate_image(pil_img):

    validator, preprocess_input, decode_predictions = load_validator()

    img = pil_img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    preds = validator.predict(arr, verbose=0)
    top20 = decode_predictions(preds, top=20)[0]

    # 1) Cari dulu apakah ada indikasi ikan di top-20
    for _, label, prob in top20:
        clean_label = label.lower().replace("_", " ")

        if clean_label in IGNORE_LABELS:
            continue

        if any(word in clean_label for word in FISH_ALLOW):
            return True, label

    # 2) Kalau tidak ada indikasi ikan, baru cek objek non-ikan
    #    tapi hanya jika confidence-nya cukup tinggi
    for _, label, prob in top20:
        clean_label = label.lower().replace("_", " ")

        if clean_label in IGNORE_LABELS:
            continue

        if prob < 0.15:
            continue

        if any(word in clean_label for word in NON_FISH_BLOCK):
            return False, label

    # 3) Default: terima gambar (jangan terlalu ketat menolak)
    return True, top20[0][1]


def preprocess_image(pil_img):

    img = pil_img.convert("RGB").resize(IMG_SIZE)

    arr = np.array(img, dtype=np.float32) / 255.0

    arr = np.expand_dims(arr, axis=0)

    return arr

def predict(model, img_array):

    pred = float(model.predict(img_array, verbose=0)[0][0])

    prob_notfresh = pred
    prob_fresh = 1 - pred

    if pred >= THRESHOLD:

        label = "Not Fresh"
        confidence = prob_notfresh

    else:

        label = "Fresh"
        confidence = prob_fresh

    return label, confidence, prob_fresh, prob_notfresh

st.markdown("""
<div class="app-header">
    <h1>Klasifikasi Kesegaran Ikan</h1>
    <p>Menggunakan metode CNN MobileNetV2 untuk mengklasifikasikan tingkat kesegaran ikan</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
background:#1e293b;
padding:20px;
border-radius:18px;
margin-bottom:20px;
box-shadow:0 2px 10px rgba(0,0,0,.05);
color:#f8fafc;
">

<h3>Informasi Sistem</h3>

<p>
Aplikasi ini digunakan untuk mengklasifikasikan tingkat kesegaran ikan nila, kembung, dan tuna berdasarkan gambar ikan yang diunggah menggunakan arsitektur MobileNetV2.
</p>

<p>
HKategori Klasifikasi:
</p>

<ul>
<li>✅ Fresh (Segar)</li>
<li>❌ Not Fresh (Tidak Segar)</li>
</ul>

<p>
Gunakan gambar ikan yang jelas agar hasil prediksi lebih akurat.
</p>

</div>
""", unsafe_allow_html=True)

with st.spinner("Memuat model..."):

    model, model_path = load_classifier()

if model is None:

    st.error(
        "Model tidak ditemukan. "
        "Pastikan file model tersedia."
    )

    st.stop()

st.subheader("Unggah Citra Ikan")

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"]
)

st.info(
    "Tips: Gunakan gambar ikan dengan pencahayaan yang baik, fokus yang jelas, dan latar belakang yang sederhana agar hasil klasifikasi lebih akurat.\n\n"
    "Catatan: Hasil klasifikasi merupakan prediksi model berdasarkan gambar ikan yang diunggah dan tidak menggantikan pemeriksaan organoleptik secara langsung."
)

if uploaded_file:

    try:

        image = Image.open(uploaded_file)

    except:

        st.error("Gambar tidak dapat dibaca.")
        st.stop()

    st.image(
        image,
        use_container_width=True
    )

    st.markdown("## Hasil Klasifikasi")

    with st.spinner("Sedang memproses gambar..."):

        valid, detected_label = validate_image(image)

    if not valid:

        st.warning(
            f"Objek tidak didukung.\n\n"
            f"Terdeteksi sebagai: {detected_label}"
        )

        st.info(
            "Silakan upload gambar ikan sesuai dataset penelitian."
        )

    else:

        img_array = preprocess_image(image)

        label, confidence, prob_fresh, prob_notfresh = predict(
            model,
            img_array
        )

        if label == "Fresh":

            st.success(
                f"✅ Segar (Fresh)\n\n"
                f"Confidence Score: {confidence*100:.2f}%"
            )

        else:

            st.error(
                f"❌ Tidak Segar (Not Fresh)\n\n"
                f"Confidence Score: {confidence*100:.2f}%"
            )

        st.markdown("### 📊 Probabilitas Klasifikasi")

        st.write(f"🟢 Fresh : {prob_fresh*100:.2f}%")
        st.progress(float(prob_fresh))

        st.write(f"🔴 Not Fresh : {prob_notfresh*100:.2f}%")
        st.progress(float(prob_notfresh))

        with st.expander("🔎 Detail Teknis"):

            st.write(f"Model : {os.path.basename(model_path)}")
            st.write("Arsitektur : MobileNetV2")
            st.write("Ukuran Input : 224 × 224")
            st.write(f"Hasil Prediksi : {label}")


st.markdown("""
<div style="
text-align:center;
padding:20px;
margin-top:40px;
color:#94a3b8;
font-size:14px;
line-height:1.7;
">

Sistem Klasifikasi Kesegaran Ikan <br>
CNN MobileNetV2 • Streamlit • Astri Salwa P M

</div>
""", unsafe_allow_html=True)
