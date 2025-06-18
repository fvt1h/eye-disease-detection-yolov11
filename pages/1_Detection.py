import streamlit as st
from ultralytics import YOLO
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer
import requests
import os
import cv2
import numpy as np

# --- PENGATURAN MODEL (WAJIB DISESUAIKAN) ---
MODEL_URL = "https://github.com/fvt1h/eye-disease-detection-yolov11/releases/download/v2.0-model/model_finetune.pt"
MODEL_FILENAME = "YOLOv11n-finetuned-roboflow.pt" 

@st.cache_resource
def load_yolo_model(url, filename):
    if not os.path.exists(filename):
        with st.spinner(f"Mengunduh model... Ini mungkin memakan waktu."):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
                st.success("Model berhasil diunduh!")
            except Exception as e:
                st.error(f"Gagal mengunduh model. Error: {e}")
                return None
    try:
        model = YOLO(filename)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model dari file. Error: {e}")
        return None

st.set_page_config(page_title="Halaman Deteksi", page_icon="🔬")
st.title("🔬 Halaman Deteksi Penyakit Mata")
st.write("Silakan pilih metode deteksi: unggah gambar atau gunakan kamera untuk deteksi real-time.")

model = load_yolo_model(MODEL_URL, MODEL_FILENAME)

if model is None:
    st.error("Aplikasi tidak bisa berjalan karena model gagal dimuat.")
    st.stop()

st.info(f"**Model yang digunakan:** Model terbaik yang telah dipilih ({MODEL_FILENAME}).")

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