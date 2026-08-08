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

html, body, [class*="css"]{
    font-family:'Plus Jakarta Sans',sans-serif!important;
}

body{
    background:#f1f5f9;
}

#MainMenu,
header,
footer{
    visibility:hidden;
}

.block-container{
    max-width:850px;
    padding-top:25px;
}

/* HEADER */

.app-header{

background:linear-gradient(135deg,#065f46,#059669);

padding:30px;

border-radius:18px;

text-align:center;

color:white;

margin-bottom:22px;

box-shadow:0 8px 20px rgba(0,0,0,.15);

}

.app-header h1{
    color:#ffffff;
    font-weight:800;
}

.app-header p{
    color:#d1fae5;
    font-size:15px;
}

/* CARD */

.info-card{

background:#1e293b;

padding:22px;

border-radius:16px;

color:white;

margin-bottom:25px;

box-shadow:0 6px 15px rgba(0,0,0,.12);

}

.info-card h2{
    color:#ffffff;
    font-weight:700;
}

.info-card p{
    color:#e2e8f0;
    line-height:1.8;
}

.info-card li{
    color:#f8fafc;
}

/* Upload */

[data-testid="stFileUploader"]{

border:2px dashed #14b8a6;

border-radius:15px;

padding:10px;

background:white;

}

/* Image */

[data-testid="stImage"] img{

border-radius:15px;

border:4px solid #14b8a6;

}

/* Alert */

[data-testid="stAlert"]{

border-radius:15px;

}

/* Progress */

.stProgress > div > div{

background:#10b981;

}

/* Expander */

details{

border-radius:12px;

padding:8px;

}

/* Footer */

.footer{
    color:#64748b;
    font-size:14px;
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
    "person", "people", "man", "woman", "boy", "girl",
    "human", "baby", "child", "portrait", "face", "head",

    # Clothes
    "shirt", "dress", "coat", "jacket", "jeans",
    "shoe", "helmet", "hat", "bag", "watch",

    # Animals
    "cat", "dog", "horse", "cow", "lion", "tiger",
    "bear", "rabbit", "bird", "snake", "frog", "monkey",

    # Vehicles
    "car", "bus", "truck", "motorcycle",
    "bicycle", "train", "airplane",

    # Objects
    "book", "chair", "table", "phone",
    "computer", "television", "monitor", "laptop",

    # Others
    "comic", "cartoon", "drawing", "poster",
    "diagram", "chart", "graph",

    # Seafood
    "shrimp", "lobster", "crab",
    "squid", "octopus"
}

IGNORE_LABELS = {
    "plate", "tray", "dish", "bowl",
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

<h1>🐟 Klasifikasi Kesegaran Ikan</h1>

<p>
Menggunakan metode CNN MobileNetV2 untuk
mengklasifikasikan tingkat kesegaran ikan
</p>

</div>
""",unsafe_allow_html=True)

st.markdown("""
<div class="info-card">

<h2>📋 Informasi Sistem</h2>

<p>

Aplikasi web ini digunakan untuk mengklasifikasikan
tingkat kesegaran ikan nila,
kembung,
dan tuna
berdasarkan gambar ikan
menggunakan arsitektur
<b>MobileNetV2</b>.

</p>

<h4 style="color:white;">Kategori Klasifikasi</h4>
<ul style="color:white;">
    <li>✅ Fresh (Segar)</li>
    <li>❌ Not Fresh (Tidak Segar)</li>
</ul>

</div>
""",unsafe_allow_html=True)
# LOAD MODEL
# ======================================================

with st.spinner("Memuat model..."):
    model, model_path = load_classifier()

if model is None:
    st.error("❌ Model tidak ditemukan.")
    st.stop()

# UPLOAD
# ======================================================

st.markdown("""
<h3 style="
color:#1e293b;
font-weight:700;
margin-bottom:10px;
">
Unggah Citra Ikan
</h3>
""", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"]
)

st.info("""

💡 Tips Penggunaan

• Gunakan pencahayaan yang baik.

• Fokus gambar harus jelas.

• Gunakan latar belakang sederhana.

• Upload JPG, JPEG atau PNG.

⚠️ Catatan

Hasil klasifikasi merupakan prediksi model
berdasarkan citra digital dan
tidak menggantikan pemeriksaan organoleptik.

""")

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
