import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2

import ssl
from urllib.request import urlopen


# Load YOLO model
@st.cache_resource
def load_model():
    ssl._create_default_https_context = ssl._create_unverified_context
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path='yolov5s.pt', force_reload=True)
    return model

# Object Detection function


def detect_objects(image, model):
    # Convert image to numpy array
    img_array = np.array(image)
    # Convert RGB to BGR format (OpenCV standard)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Perform inference
    results = model(img_array)
    # Get detection results
    # Render the detected results on the image
    results_img = np.squeeze(results.render())

    return results_img


# Streamlit UI
st.title("YOLO Object Detection App")
st.write("Upload an image to perform object detection using a trained YOLO model.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image using PIL
    image = Image.open(uploaded_file)

    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Processing...")

    # Load model
    model = load_model()

    # Perform object detection
    detected_img = detect_objects(image, model)

    # Convert BGR to RGB for displaying with Streamlit
    detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)

    # Display detected image
    st.image(detected_img, caption="Detected Image", use_column_width=True)
