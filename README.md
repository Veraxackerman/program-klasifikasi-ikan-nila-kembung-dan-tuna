# Klasifikasi Kesegaran Ikan — Streamlit App

Aplikasi web untuk mendeteksi kesegaran ikan (Fresh / Not Fresh) menggunakan
model CNN MobileNetV2 yang dilatih dengan penanganan class imbalance
(`compute_class_weight`).

## Struktur Folder

```
streamlit_app/
├── app.py                 # Aplikasi utama Streamlit
├── requirements.txt        # Dependency Python
├── model/
│   └── model_final.keras   # Model hasil training (WAJIB ditambahkan manual)
└── README.md
```

## 1. Menyiapkan Model

File model **tidak disertakan** di sini karena ukurannya besar. Ambil dari
folder `output/` hasil menjalankan notebook `klasifikasi_ikan_classweight.ipynb`:

- `output/model_final.keras` (disarankan), atau
- `output/model_final.h5`

Copy salah satu file tersebut ke folder `model/` di project ini.
Kalau pakai `.h5`, ubah baris `MODEL_PATH` di `app.py`:

```python
MODEL_PATH = os.path.join('model', 'model_final.h5')
```

## 2. Menjalankan di Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501` di browser.

## 3. Deploy ke Streamlit Community Cloud

### a. Push ke GitHub
Buat repository baru (public atau private) berisi:
- `app.py`
- `requirements.txt`
- folder `model/` beserta file model-nya

> **Catatan ukuran file:** GitHub membatasi file biasa ke 100 MB. Kalau model
> lebih besar dari itu, pakai [Git LFS](https://git-lfs.com/) atau simpan model
> di Google Drive/Hugging Face Hub lalu unduh otomatis saat app start (lihat
> bagian 4).

### b. Deploy
1. Buka [share.streamlit.io](https://share.streamlit.io) dan login pakai akun GitHub.
2. Klik **New app** → pilih repository, branch, dan `app.py` sebagai entry point.
3. Klik **Deploy**. Streamlit akan otomatis membaca `requirements.txt` dan
   menginstall semua dependency.
4. Tunggu proses build selesai (biasanya 3–5 menit karena TensorFlow cukup besar).

### c. Update App yang Sudah Ada
Kalau ini update untuk app yang sudah live di
`klasifikasi-kesegaran-ikan.streamlit.app`, cukup:
```bash
git add .
git commit -m "Update model dengan class_weight"
git push
```
Streamlit Cloud otomatis redeploy setiap ada push ke branch yang terhubung.

## 4. (Opsional) Model Besar via Download Otomatis

Kalau model tidak muat di GitHub, tambahkan kode di awal `app.py` untuk
mengunduh model dari Google Drive/Hugging Face saat pertama kali dijalankan,
lalu simpan di cache lokal Streamlit Cloud. Beri tahu saya kalau mau dibuatkan
kode ini — perlu link download langsung ke file model kamu.

## 5. Kustomisasi Lanjutan

- **`validate_image()`** di `app.py` masih placeholder (selalu menerima
  gambar). Kalau app lama sudah punya validator berbasis MobileNetV2 ImageNet
  untuk menolak foto non-ikan, fungsi itu bisa disalin ke sini.
- Ganti threshold 0.5 di fungsi `predict()` kalau mau lebih ketat/longgar.
