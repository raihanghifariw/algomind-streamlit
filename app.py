import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import ssl

# Memastikan SSL untuk menghindari masalah koneksi saat download model
ssl._create_default_https_context = ssl._create_unverified_context

# Menggunakan cache untuk memuat model hanya sekali


@st.cache_resource
def load_model():
    # Menggunakan YOLOv5 pre-trained
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

# Fungsi deteksi objek pada gambar


def detect_objects(image, model):
    # Konversi gambar ke numpy array dan ke BGR untuk kompatibilitas OpenCV
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Lakukan inferensi deteksi objek
    results = model(img_array)
    results_img = np.squeeze(results.render())  # Render hasil pada gambar

    # Konversi BGR kembali ke RGB untuk tampilan di Streamlit
    detected_img = cv2.cvtColor(results_img, cv2.COLOR_BGR2RGB)
    return detected_img


# Tampilan UI Streamlit
st.title("YOLO Object Detection App")
st.write("Unggah gambar untuk melakukan deteksi objek menggunakan model YOLO.")

# File uploader
uploaded_file = st.file_uploader(
    "Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membuka gambar menggunakan PIL
    image = Image.open(uploaded_file)

    # Tampilkan gambar yang diunggah
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)
    st.write("Memproses...")

    # Muat model YOLO
    model = load_model()

    # Deteksi objek pada gambar
    detected_img = detect_objects(image, model)

    # Tampilkan hasil deteksi
    st.image(detected_img, caption="Hasil Deteksi", use_column_width=True)
