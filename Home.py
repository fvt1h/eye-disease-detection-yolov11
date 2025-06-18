import streamlit as st
from PIL import Image
import os

# Menggunakan cache agar pemuatan aset (teks dan gambar) lebih cepat
@st.cache_data
def load_asset(path, asset_type='text'):
    """
    Fungsi generik untuk memuat teks atau gambar dari path.
    """
    if not os.path.exists(path):
        st.warning(f"Aset tidak ditemukan di: {path}")
        return None
    try:
        if asset_type == 'text':
            with open(path, "r", encoding="utf-8") as file:
                return file.read()
        elif asset_type == 'image':
            return Image.open(path)
    except Exception as e:
        st.error(f"Gagal memuat aset: {e}")
        return None

# Konfigurasi Halaman (Judul di tab browser dan ikon)
st.set_page_config(
    page_title="Deteksi Penyakit Mata",
    page_icon="👁️"
)

# Konten Halaman Beranda
st.title("👁️ Deteksi Penyakit Mata (Glaucoma & Diabetic Retinopathy)")
st.write(
    "Selamat datang di aplikasi deteksi penyakit mata berbasis AI. "
    "Aplikasi ini menggunakan model deep learning untuk mengidentifikasi potensi Glaucoma "
    "dan Diabetic Retinopathy dari gambar fundus mata. Silakan pilih halaman di sidebar untuk memulai."
)
st.info("Pilih halaman **Detection** untuk mencoba model atau halaman **Code Explanation** untuk detail teknis.")

st.divider()

st.header("Informasi Penyakit")
st.write("Berikut adalah informasi singkat mengenai kondisi mata yang menjadi fokus deteksi.")

# Siapkan data informasi untuk setiap kondisi
disease_info = {
    "Diabetic Retinopathy": {
        "image_path": "assets/sample_images/diabetic_retinopathy_example.jpg",
        "explanation_path": "assets/explanations/diabetic_retinopathy.txt"
    },
    "Glaucoma": {
        "image_path": "assets/sample_images/glaucoma_example.jpg",
        "explanation_path": "assets/explanations/glaucoma.txt"
    },
    "Normal": {
        "image_path": "assets/sample_images/normal_example.jpg",
        "explanation_path": "assets/explanations/normal.txt"
    }
}

# Loop untuk menampilkan setiap penyakit dengan layout vertikal
for title, data in disease_info.items():
    st.divider() # Beri garis pemisah di atas setiap bagian
    st.subheader(title)
    
    # Muat aset gambar dan teks
    image = load_asset(data["image_path"], 'image')
    explanation = load_asset(data["explanation_path"], 'text')

    # Tampilkan gambar terlebih dahulu
    if image:
        st.image(image, use_container_width=True) # use_container_width agar lebar gambar pas dengan halaman
    
    # Tampilkan penjelasan di bawah gambar
    if explanation:
        st.write(explanation)