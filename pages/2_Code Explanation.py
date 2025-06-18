import streamlit as st
from PIL import Image
import os

@st.cache_data
def load_image(path):
    if not os.path.exists(path):
        st.warning(f"File gambar tidak ditemukan di: {path}")
        return None
    try:
        return Image.open(path)
    except Exception as e:
        st.error(f"Gagal memuat gambar: {e}")
        return None

st.set_page_config(page_title="Informasi Model", page_icon="⚙️")
st.title("⚙️ Informasi Model & Proses Training")
st.write("Halaman ini berisi detail teknis mengenai model AI yang digunakan, bukti performanya, serta gambaran umum proses training.")

st.header("Bukti Performa Model")
with st.expander("Lihat Hasil Training & Evaluasi Model Terbaik"):
    base_path = "assets/training_results"
    
    st.subheader("Ringkasan Metrik Evaluasi")
    eval_img = load_image(f"{base_path}/eval_metrics_summary.png")
    if eval_img: st.image(eval_img, caption="Tabel metrik mAP, Precision, dan Recall pada data validasi.", use_container_width=True)

    st.subheader("Confusion Matrix")
    cm_img = load_image(f"{base_path}/confusion_matrix_normalized.png")
    if cm_img: st.image(cm_img, caption="Confusion matrix menunjukkan akurasi model per kelas.", use_container_width=True)

    st.subheader("Grafik Performa Selama Training")
    results_img = load_image(f"{base_path}/results.png")
    if results_img: st.image(results_img, caption="Grafik peningkatan performa model selama proses training.", use_container_width=True)
    
    st.subheader("Validation Batch Prediction")
    results_img = load_image(f"{base_path}/val_batch2_pred.jpg")
    if results_img: st.image(results_img, caption="Hasil prediksi dari data validation.", use_container_width=True)

st.divider()

st.header("Alur Kode Lengkap Proses Training")
st.info("Berikut merupakan kode yang dijalankan di Google Colab dan di Kaggle untuk melatih model.")

# Struktur data untuk menyimpan setiap sel kode dan penjelasannya
training_workflow = [
    {
        "title": "Sel 1: Instalasi & Impor Library",
        "code": """# Instalasi library yang dibutuhkan
!pip install -q roboflow ultralytics

# Impor library
import os
import shutil
import yaml
import random
import torch
from datetime import datetime
from ultralytics import YOLO
import zipfile
from google.colab import drive
from kaggle_secrets import UserSecretsClient # Tergantung lingkungan""",
        "explanation": "Sel pertama ini bertujuan untuk menyiapkan semua 'alat bantu' (library) yang diperlukan. `!pip install` digunakan untuk menginstal library, dan `import` digunakan untuk memuatnya ke dalam skrip agar fungsinya bisa kita gunakan."
    },
    {
        "title": "Sel 2: Fungsi `seed_everything`",
        "code": """def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False""",
        "explanation": "Fungsi ini dibuat untuk memastikan proses training bisa diulang (reproducible). Dengan mengatur 'seed' atau angka acak awal yang sama, hasil training akan konsisten setiap kali dijalankan, sehingga memudahkan untuk perbandingan antar eksperimen."
    },
    {
        "title": "Sel 3: Persiapan Data & Pengaturan Utama",
        "code": """# Menghubungkan ke Google Drive (untuk Colab)
drive.mount('/content/drive')

# Pengaturan utama
GLOBAL_SEED = 42
BASE_MODEL_NAME = "yolov11m.pt"
IMAGE_SIZE = 640

# Mengunduh dataset dari Roboflow menggunakan API Key
# (Kode untuk otentikasi dan download diletakkan di sini)
# ...
dataset = version.download("yolov11")

# Mendefinisikan path input dan output
BASE_DATA_YAML_PATH = os.path.join(dataset.location, "data.yaml")
PROJECT_OUTPUT_PARENT_DIR = "/content/drive/MyDrive/Skripsi_AI_Outputs"

# Menjalankan seed dan membuat folder output
seed_everything(GLOBAL_SEED)
os.makedirs(PROJECT_OUTPUT_PARENT_DIR, exist_ok=True)""",
        "explanation": "Sel ini adalah pusat kendali. Pertama, ia menghubungkan ke Google Drive untuk penyimpanan permanen. Kemudian, ia mendefinisikan parameter global seperti model mana yang akan dipakai. Setelah itu, ia mengunduh dataset yang sudah diaugmentasi dari Roboflow. Terakhir, ia mendefinisikan path input (`BASE_DATA_YAML_PATH`) dan path output ke Google Drive (`PROJECT_OUTPUT_PARENT_DIR`)."
    },
    {
        "title": "Sel 4: Phase 1 - Initial Training",
        "code": """# Membuat folder output unik untuk Phase 1
phase1_run_output_dir = os.path.join(...)

# Menentukan parameter Phase 1
phase1_epochs = 70
phase1_optimizer = "Adam"
# ...

# Memulai training
model_p1 = YOLO(BASE_MODEL_NAME)
model_p1.train(
    data=BASE_DATA_YAML_PATH,
    epochs=phase1_epochs,
    optimizer=phase1_optimizer,
    project=phase1_run_output_dir,
    name="results"
)

# Mengambil path model terbaik dari Phase 1
path_best_model_phase1 = os.path.join(...)""",
        "explanation": "Ini adalah tahap training awal. Model dilatih dengan parameter yang dirancang untuk pembelajaran cepat (Optimizer Adam, learning rate yang cukup tinggi). Tujuannya adalah agar model dapat mengenali fitur-fitur umum dari dataset secara luas. Model terbaik dari fase ini akan disimpan untuk digunakan di fase selanjutnya."
    },
    {
        "title": "Sel 5: Phase 2 - Fine-Tuning",
        "code": """# Cek apakah model dari Phase 1 ada, lalu lanjutkan
if path_best_model_phase1:
    # Membuat folder output unik untuk Phase 2
    phase2_run_output_dir = os.path.join(...)
    
    # Menentukan parameter Phase 2 (berbeda dari Phase 1)
    phase2_epochs = 35
    phase2_optimizer = "SGD"
    phase2_lr0 = 0.0001
    # ...

    # Memulai training dari model terbaik Phase 1
    model_p2 = YOLO(path_best_model_phase1)
    model_p2.train(...)

    # Mengambil path model terbaik dari Phase 2
    path_best_model_phase2 = os.path.join(...)
""",
        "explanation": "Ini adalah tahap penyempurnaan (fine-tuning). Model terbaik dari Phase 1 dilatih kembali, tetapi dengan 'sentuhan yang lebih halus'—menggunakan learning rate yang jauh lebih kecil dan optimizer SGD. Tujuannya adalah untuk menyesuaikan model secara presisi dengan dataset dan meningkatkan akurasinya."
    },
    {
        "title": "Sel 6: Evaluasi & Zipping",
        "code": """# Evaluasi model final pada validation set
evaluation_model = YOLO(final_model_path)
evaluation_model.val(data=BASE_DATA_YAML_PATH, split='val')

# Mengemas hasil ke dalam file .zip
zip_path = f"{run_output_dir}_Outputs.zip"
with zipfile.ZipFile(zip_path, 'w') as zipf:
    # ... (logika untuk memasukkan file output dan model .pt) ...
""",
        "explanation": "Setelah training selesai, model terbaik dievaluasi pada data validasi untuk mengukur performa akhirnya. Kemudian, semua file hasil (grafik, log, dan model .pt itu sendiri) dikemas ke dalam sebuah file ZIP agar mudah diunduh dan diarsipkan."
    }
]

# Loop untuk menampilkan setiap langkah di Streamlit
for step in training_workflow:
    st.subheader(step["title"])
    st.code(step["code"], language="python")
    with st.expander("Lihat Penjelasan untuk Sel Ini"):
        st.write(step["explanation"])
    st.divider()