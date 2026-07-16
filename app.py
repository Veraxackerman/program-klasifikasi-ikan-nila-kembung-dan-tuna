import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# ============================================================
# KONFIGURASI
# ============================================================
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Fresh', 'Not_Fresh']
MODEL_PATH = os.path.join('model', 'model_final.keras')  # ganti ke .h5 kalau pakai itu

st.set_page_config(
    page_title="Klasifikasi Kesegaran Ikan",
    page_icon="🐟",
    layout="centered"
)


# ============================================================
# LOAD MODEL (di-cache supaya tidak reload tiap interaksi)
# ============================================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


# ============================================================
# VALIDASI GAMBAR (opsional — cek apakah gambar mengandung ikan)
# Sesuaikan/lengkapi fungsi ini kalau kamu sudah punya validator
# dari app sebelumnya (mis. berbasis MobileNetV2 ImageNet).
# ============================================================
def validate_image(pil_img):
    """
    Placeholder validator. Kembalikan True kalau gambar dianggap layak
    diproses. Default: selalu terima gambar (tidak menolak apa pun).
    """
    return True


# ============================================================
# PREDIKSI
# ============================================================
def predict(model, pil_img):
    img = pil_img.convert('RGB').resize(IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    prob = model.predict(arr, verbose=0)[0][0]

    # sigmoid output = probabilitas kelas Not_Fresh (index 1)
    label = 'Not_Fresh' if prob >= 0.5 else 'Fresh'
    confidence = prob if prob >= 0.5 else 1 - prob

    return label, float(confidence)


# ============================================================
# UI
# ============================================================
st.title("🐟 Klasifikasi Kesegaran Ikan")
st.write(
    "Unggah foto ikan untuk mendeteksi apakah ikan tersebut "
    "**segar (Fresh)** atau **tidak segar (Not Fresh)** "
    "menggunakan model CNN MobileNetV2."
)

uploaded_file = st.file_uploader(
    "Pilih gambar ikan (jpg/jpeg/png)",
    type=['jpg', 'jpeg', 'png', 'bmp', 'webp']
)

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)
    st.image(pil_img, caption="Gambar yang diunggah", use_container_width=True)

    if not validate_image(pil_img):
        st.error(
            "Gambar tidak terdeteksi sebagai ikan. "
            "Silakan unggah foto ikan yang lebih jelas."
        )
    else:
        with st.spinner("Menganalisis gambar..."):
            model = load_model()
            label, confidence = predict(model, pil_img)

        if label == 'Fresh':
            st.success(f"✅ Ikan **SEGAR** (confidence: {confidence*100:.2f}%)")
        else:
            st.error(f"⚠️ Ikan **TIDAK SEGAR** (confidence: {confidence*100:.2f}%)")

        st.progress(min(max(confidence, 0.0), 1.0))

st.markdown("---")
st.caption(
    "Model: MobileNetV2 (fine-tuned) — dilatih dengan penanganan "
    "class imbalance menggunakan `compute_class_weight`."
)
