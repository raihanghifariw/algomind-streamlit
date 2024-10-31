import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import ssl

# Memastikan SSL untuk menghindari masalah koneksi saat download model
ssl._create_default_https_context = ssl._create_unverified_context


@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model


def detect_objects(image, model):
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    results = model(img_array)
    results_img = np.squeeze(results.render())
    detected_img = cv2.cvtColor(results_img, cv2.COLOR_BGR2RGB)
    return detected_img


def show():
    st.title("YOLO Object Detection App")
    st.write("Unggah gambar untuk melakukan deteksi objek menggunakan model YOLO.")

    uploaded_file = st.file_uploader(
        "Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)
        st.write("Memproses...")

        model = load_model()
        detected_img = detect_objects(image, model)
        st.image(detected_img, caption="Hasil Deteksi", use_column_width=True)
