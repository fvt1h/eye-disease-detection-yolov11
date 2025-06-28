import streamlit as st
from ultralytics import YOLO
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer
import os
import cv2
import numpy as np


MODEL_PATH = "models/best_yoloV11n.pt"

@st.cache_resource
def load_yolo_model(model_path):
    """
    Memuat model YOLO dari path lokal yang diberikan.
    Fungsi ini di-cache agar pemuatan hanya terjadi sekali.
    """
    # Cek apakah file model ada di path yang ditentukan.
    if not os.path.exists(model_path):
        st.error(f"File model tidak ditemukan di path: {model_path}")
        st.info("Pastikan Anda sudah meletakkan file model (.pt) di dalam folder 'models'.")
        return None
    
    # Langsung muat model dari file lokal.
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model dari file. Error: {e}")
        return None

# --- UI HALAMAN PREDIKSI ---
st.set_page_config(page_title="Halaman Deteksi", page_icon="🔬")
st.title("🔬 Halaman Deteksi Penyakit Mata")
st.write("Silakan pilih metode deteksi: unggah gambar atau gunakan kamera untuk deteksi real-time.")

# Memuat model dari path lokal yang sudah ditentukan di atas.
model = load_yolo_model(MODEL_PATH)

# Hentikan eksekusi jika model gagal dimuat.
if model is None:
    st.stop()

st.info(f"**Model yang digunakan:** {os.path.basename(MODEL_PATH)}")

# --- Sisa kode untuk tab dan deteksi tetap sama seperti sebelumnya ---
tab1, tab2 = st.tabs(["🖼️ Unggah Gambar", "📹 Deteksi Real-time (Webcam)"])

with tab1:
    st.header("Deteksi dari Gambar yang Diunggah")
    uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang Diunggah", width=400)
        
        if st.button("Mulai Deteksi pada Gambar"):
            with st.spinner("Model sedang bekerja..."):
                results = model.predict(image)
                annotated_image = results[0].plot()
                annotated_image_rgb = annotated_image[..., ::-1]
                st.image(annotated_image_rgb, caption="Hasil Deteksi", width=400)

with tab2:
    st.header("Deteksi Langsung dari Webcam")
    st.write("Klik 'START' di bawah untuk mengaktifkan kamera.")
    
    class VideoProcessor:
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            mirrored_img = cv2.flip(img, 1)
            results = model.predict(mirrored_img, verbose=False)
            annotated_frame = results[0].plot()
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    webrtc_streamer(
        key="detection-webcam",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )