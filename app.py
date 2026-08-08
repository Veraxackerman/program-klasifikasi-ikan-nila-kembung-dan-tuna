import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# KONFIGURASI HALAMAN
# ======================================================

st.set_page_config(
    page_title="Sistem Klasifikasi Kesegaran Ikan",
    page_icon="🐟",
    layout="centered"
)

# CSS
# ======================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html,
body,
[class*="css"]{
    font-family:'Plus Jakarta Sans',sans-serif !important;
}

body{
    background:#111827;
}

#MainMenu,
footer,
header{
    visibility:hidden;
}

.block-container{
    max-width:900px;
    padding-top:2rem;
}

.app-header{
    background:linear-gradient(135deg,#064e3b,#065f46);
    color:white;
    padding:2rem;
    border-radius:18px;
    text-align:center;
    margin-bottom:1.5rem;
}

.app-header h1{
    font-size:2rem;
    font-weight:800;
    margin-bottom:.3rem;
}

.app-header p{
    color:rgba(255,255,255,.85);
    margin:0;
}

[data-testid="stImage"] img{
    border-radius:14px;
}

[data-testid="stFileUploader"]{
    border:2px dashed #475569;
    border-radius:16px;
    background:#1e293b;
    padding:8px;
}

@media(max-width:768px){

.app-header{
    padding:1.5rem;
}

.app-header h1{
    font-size:1.6rem;
}

.block-container{
    padding-left:1rem;
    padding-right:1rem;
}

}

</style>
""", unsafe_allow_html=True)

# KONSTANTA
# ======================================================

IMG_SIZE = (224, 224)
THRESHOLD = 0.5

# LABEL IKAN
# ======================================================

FISH_ALLOW = {
    "fish", "tilapia", "catfish", "carp", "goldfish", "tench", "pike",
    "salmon", "tuna", "mackerel", "cod", "trout", "snapper", "grouper",
    "flounder", "sole", "halibut", "barracuda", "marlin", "swordfish",
    "anchovy", "herring", "sardine", "coho", "eel", "shark", "ray",
    "stingray", "lionfish", "puffer", "blowfish"
}

# LABEL BUKAN IKAN
# ======================================================

NON_FISH_BLOCK = {

    # Human
    ""person", "people", "man", "woman", "boy", "girl",
    "human", "baby", "child", "portrait", "face", "head",

    # Clothes
    "shirt", "dress", "coat", "jacket", "jeans",
    "shoe", "helmet", "hat", "bag", "watch",

    # Animals
    "cat", "dog", "horse", "cow", "lion", "tiger",
    "bear", "rabbit", "bird", "snake", "frog", "monkey",

    # Vehicles
    ""car", "bus", "truck", "motorcycle",
    "bicycle", "train", "airplane",

    # Objects
    ""book", "chair", "table", "phone",
    "computer", "television", "monitor", "laptop",

    # Others
    "comic", "cartoon", "drawing", "poster",
    "diagram", "chart", "graph",

    # Seafood
    "shrimp", "lobster", "crab",
    "squid", "octopus"
}

IGNORE_LABELS = {
    " "plate", "tray", "dish", "bowl",
    "platter", "menu", "table", "napkin"
}

# LOAD MODEL KLASIFIKASI
# ======================================================

@st.cache_resource
def load_classifier():

    model_paths = [
        "model_final.keras",
        "output/model_final.keras",
        "model_final.h5"
    ]

    for path in model_paths:

        if os.path.exists(path):

            try:

                model = tf.keras.models.load_model(
                    path,
                    compile=False
                )

                return model, path

            except Exception as e:

                st.error(f"Gagal memuat model:\n{e}")

    return None, None

# LOAD VALIDATOR IMAGENET
# ======================================================

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

# VALIDASI GAMBAR
# ======================================================

def validate_image(pil_img):

    try:

        validator, preprocess_input, decode_predictions = load_validator()

        img = pil_img.convert("RGB").resize((224, 224))

        arr = np.array(img, dtype=np.float32)

        arr = preprocess_input(arr)

        arr = np.expand_dims(arr, axis=0)

        preds = validator.predict(arr, verbose=0)

        top20 = decode_predictions(preds, top=20)[0]

        # --------------------------
        # Cari label ikan
        # --------------------------

        for _, label, prob in top20:

            clean = label.lower().replace("_", " ")

            if clean in IGNORE_LABELS:
                continue

            if any(word in clean for word in FISH_ALLOW):
                return True, label

        # --------------------------
        # Cari objek non ikan
        # --------------------------

        for _, label, prob in top20:

            clean = label.lower().replace("_", " ")

            if clean in IGNORE_LABELS:
                continue

            if prob < 0.15:
                continue

            if any(word in clean for word in NON_FISH_BLOCK):
                return False, label

        return True, top20[0][1]

    except Exception as e:

        st.warning(
            "Validator ImageNet tidak dapat dijalankan.\n"
            "Validasi objek dilewati."
        )

        return True, "Unknown"

# PREPROCESSING
# ======================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(IMG_SIZE)

    img_array = np.array(
        image,
        dtype=np.float32
    )

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array

# PREDIKSI
# ======================================================

def predict(model, img_array):

    pred = float(
        model.predict(
            img_array,
            verbose=0
        )[0][0]
    )

    prob_notfresh = pred

    prob_fresh = 1 - pred

    if pred >= THRESHOLD:

        label = "Not Fresh"

        confidence = prob_notfresh

    else:

        label = "Fresh"

        confidence = prob_fresh

    return (
        label,
        confidence,
        prob_fresh,
        prob_notfresh
    )

# HEADER
# ======================================================

st.markdown("""
<div class="app-header">
<h1>Klasifikasi Kesegaran Ikan</h1>
<p>
Menggunakan CNN MobileNetV2
untuk mengklasifikasikan
tingkat kesegaran ikan.
</p>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div style="
background:#1e293b;
padding:20px;
border-radius:18px;
margin-bottom:20px;
color:white;
">

<h3>Informasi Sistem</h3>

<p>
Aplikasi ini digunakan untuk
mengklasifikasikan tingkat
kesegaran ikan nila,
kembung,
dan tuna
berdasarkan citra digital.
</p>

<b>Kategori:</b>

<ul>
<li>✅ Fresh</li>
<li>❌ Not Fresh</li>
</ul>

</div>
""", unsafe_allow_html=True)

# LOAD MODEL
# ======================================================

with st.spinner("Memuat model..."):
    model, model_path = load_classifier()

if model is None:
    st.error("❌ Model tidak ditemukan.")
    st.stop()

# UPLOAD
# ======================================================

st.subheader("Unggah Citra Ikan")

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"]
)

st.info(
    "Tips: Gunakan gambar ikan dengan pencahayaan yang baik, fokus yang jelas, "
    "dan latar belakang sederhana agar hasil klasifikasi lebih akurat.\n\n"
    "Catatan: Hasil klasifikasi merupakan prediksi model dan tidak "
    "menggantikan pemeriksaan organoleptik."
)

# PREDIKSI
# ======================================================

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

    except Exception as e:

        st.exception(e)
        st.stop()

    st.image(
        image,
        caption="Gambar yang diunggah",
        width=450
    )

    st.markdown("## Hasil Klasifikasi")

    with st.spinner("Sedang memproses gambar..."):

        try:

            valid, detected_label = validate_image(image)

        except Exception as e:

            st.exception(e)
            st.stop()

    if not valid:

        st.warning(
            f"Objek tidak didukung.\n\n"
            f"Terdeteksi sebagai: {detected_label}"
        )

        st.info(
            "Silakan unggah gambar ikan sesuai dataset penelitian."
        )

        st.stop()

    img_array = preprocess_image(image)

    try:

        label, confidence, prob_fresh, prob_notfresh = predict(
            model,
            img_array
        )

    except Exception as e:

        st.exception(e)
        st.stop()

    if label == "Fresh":

        st.success(
            f"✅ Fresh\n\n"
            f"Confidence Score : {confidence*100:.2f}%"
        )

    else:

        st.error(
            f"❌ Not Fresh\n\n"
            f"Confidence Score : {confidence*100:.2f}%"
        )

    st.markdown("### 📊 Probabilitas")

    st.write(f"🟢 Fresh : {prob_fresh*100:.2f}%")
    st.progress(float(prob_fresh))

    st.write(f"🔴 Not Fresh : {prob_notfresh*100:.2f}%")
    st.progress(float(prob_notfresh))

    with st.expander("🔎 Detail Teknis"):

        st.write(
            f"Model : {os.path.basename(model_path)}"
        )

        st.write("Arsitektur : MobileNetV2")

        st.write("Input : 224 × 224")

        st.write(f"Hasil Prediksi : {label}")

        st.write(f"Confidence : {confidence*100:.2f}%")

# FOOTER
# ======================================================

st.markdown(
"""
<div style="
text-align:center;
padding:25px;
margin-top:40px;
color:#94a3b8;
font-size:14px;
">

<b>Sistem Klasifikasi Kesegaran Ikan</b><br>

MobileNetV2 Streamlit © 2026 Astri Salwa Putri Madani

</div>
""",
unsafe_allow_html=True
)
