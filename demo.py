# # import streamlit as st
# # import torch
# # import numpy as np
# # import cv2
# # from PIL import Image
# # import ssl

# # # Memastikan SSL untuk menghindari masalah koneksi saat download model
# # ssl._create_default_https_context = ssl._create_unverified_context


# # @st.cache_resource
# # def load_model():
# #     model = torch.hub.load('ultralytics/yolov5',
# #                            'yolov5s', pretrained=True)
# #     return model


# # def detect_objects(image, model):
# #     img_array = np.array(image)
# #     img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
# #     results = model(img_array)
# #     results_img = np.squeeze(results.render())
# #     detected_img = cv2.cvtColor(results_img, cv2.COLOR_BGR2RGB)
# #     return detected_img


# # def show():

# #     st.title("YOLO Object Detection App")
# #     st.write("Unggah gambar untuk melakukan deteksi objek menggunakan model YOLO.")

# #     uploaded_file = st.file_uploader(
# #         "Pilih gambar...", type=["jpg", "jpeg", "png"])

# #     if uploaded_file is not None:
# #         image = Image.open(uploaded_file)
# #         st.image(image, caption="Gambar yang diunggah", use_column_width=True)
# #         st.write("Memproses...")

# #         model = load_model()
# #         detected_img = detect_objects(image, model)
# #         st.image(detected_img, caption="Hasil Deteksi", use_column_width=True)
# import streamlit as st
# import requests
# from PIL import Image
# import torch
# import numpy as np
# import cv2

# import ssl
# from urllib.request import urlopen


# # Load YOLO model
# @st.cache_resource
# def load_model():
#     ssl._create_default_https_context = ssl._create_unverified_context
#     model = torch.hub.load('ultralytics/yolov5', 'custom',
#                            path='yolov5s.pt', force_reload=True)
#     return model

# # Object Detection function


# def detect_objects(image, model):
#     # Convert image to numpy array
#     img_array = np.array(image)
#     # Convert RGB to BGR format (OpenCV standard)
#     img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

#     # Perform inference
#     results = model(img_array)
#     # Get detection results
#     # Render the detected results on the image
#     results_img = np.squeeze(results.render())

#     return results_img


# # Streamlit UI
# st.title("YOLO Object Detection App")
# st.write("Upload an image to perform object detection using a trained YOLO model.")

# # Upload image
# uploaded_file = st.file_uploader(
#     "Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Open image using PIL
#     image = Image.open(uploaded_file)

#     # Display uploaded image
#     st.image(image, caption="Uploaded Image", use_column_width=True)
#     st.write("")
#     st.write("Processing...")

#     # Load model
#     model = load_model()

#     # Perform object detection
#     detected_img = detect_objects(image, model)

#     # Convert BGR to RGB for displaying with Streamlit
#     detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)

#     # Display detected image
#     st.image(detected_img, caption="Detected Image", use_column_width=True)

# Import necessary libraries
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import product
import torch
from WD3QNE_deepQnet import WD3QNE

# Load pre-saved files
DATA_PATH = "main/data/"
AGENT_ACTIONS_FILE = DATA_PATH + "/WD3QNE-algorithm/agent_actionsb.npy"
PHYS_ACTIONS_FILE = DATA_PATH + "/WD3QNE-algorithm/phys_actionsb.npy"
CSV_FILE = DATA_PATH + "/WD3QNE-algorithm/rl_test_data_final_cont.csv"
MODEL_FILE = DATA_PATH + "/WD3QNE-algorithm/dist_noW100.pt"

# Load data and predictions
data = pd.read_csv(CSV_FILE)
agent_actions = np.load(AGENT_ACTIONS_FILE)
phys_actions = np.load(PHYS_ACTIONS_FILE)

# Load model


def load_model(model_path, state_dim, n_actions, ensemble_size):
    model = WD3QNE(state_dim=state_dim, num_actions=n_actions,
                   ensemble_size=ensemble_size)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    for i, net in enumerate(model.Q_ensemble):
        net.load_state_dict(checkpoint[f'Q_ensemble_{i}'])
    return model


state_dim = 37  # Sesuaikan dengan konfigurasi model
n_actions = 25
ensemble_size = 5
model = load_model(MODEL_FILE, state_dim, n_actions, ensemble_size)

# Page title
st.title("WD3QNE Action Prediction Demo")
st.divider()
st.sidebar.image("main/assets/sepvisor.jpeg")  # Update with your logo path

# Introduction
st.write("This page demonstrates the WD3QNE model for predicting actions based on ICU patient data.")

options = data['patient_id'].unique()
selected_option = st.selectbox("Select a Patient ID:", options)

save_indexes = data[data['patient_id'] == selected_option].index.tolist()
st.write(f"You selected: {selected_option}")
st.divider()

# Display selected patient's data
st.subheader("Selected Patient's Condition in ICU")
selected_param = ["SOFA", "GCS", "SIRS", "Temp_C", "SysBP", "DiaBP", "HR"]
show_table = data.loc[save_indexes, selected_param].reset_index(drop=True)

plt.figure(figsize=(10, 6))
for param in selected_param:
    plt.plot(show_table[param], label=param)

plt.xlabel('4-Hourly Timestamps')
plt.ylabel('Normalized Values')
plt.title('Parameters Over Time')
plt.legend()
plt.grid(True)
st.pyplot(plt)
plt.close()

# Predict button
st.subheader("Make Treatment Recommendation With AI")
if st.button("Predict"):
    st.subheader("Dose Intensity Recommendation for the Patient")
    phys_act, ai_act = [], []
    doses_intensity = ["Zero", "Low", "Medium", "High", "Very High"]
    pair_of_act = list(product(doses_intensity, repeat=2))

    for i in save_indexes:
        phys_act.append(pair_of_act[phys_actions[i]])
        ai_act.append(pair_of_act[agent_actions[i]])

        temp = {
            'IV Fluid': {'Physician': phys_act[-1][0], 'AI': ai_act[-1][0]},
            'Vasopressor': {'Physician': phys_act[-1][1], 'AI': ai_act[-1][1]}
        }
        df = pd.DataFrame(temp)
        st.table(df)
        st.divider()

    # Differences in actions
    st.subheader("Differences in actions taken between Physician and AI")
    col1, col2 = st.columns(2)

    with col1:
        y_labels = ["Zero", "Low", "Medium", "High", "Very High"]
        y_values_phys = [y_labels.index(act[0]) for act in phys_act]
        y_values_ai = [y_labels.index(act[0]) for act in ai_act]

        plt.plot(range(len(phys_act)), y_values_phys,
                 label='Physician', marker='o', linestyle='-')
        plt.plot(range(len(ai_act)), y_values_ai,
                 label='AI', marker='o', linestyle='-')
        plt.xlabel('4-Hourly Timestamps')
        plt.ylabel('Intensity')
        plt.title('IV Fluid Dose Intensity')
        plt.yticks(range(len(y_labels)), y_labels)
        plt.legend()
        st.pyplot(plt)

    with col2:
        y_values_phys = [y_labels.index(act[1]) for act in phys_act]
        y_values_ai = [y_labels.index(act[1]) for act in ai_act]

        plt.plot(range(len(phys_act)), y_values_phys,
                 label='Physician', marker='o', linestyle='-')
        plt.plot(range(len(ai_act)), y_values_ai,
                 label='AI', marker='o', linestyle='-')
        plt.xlabel('4-Hourly Timestamps')
        plt.ylabel('Intensity')
        plt.title('Vasopressor Dose Intensity')
        plt.yticks(range(len(y_labels)), y_labels)
        plt.legend()
        st.pyplot(plt)

# Footer
text = "\u00a92024 HFR Company. All rights reserved. The content on this website is protected by copyright law."
text2 = "For permission requests, please contact +6287870190448."
centered_text = f"<p style='text-align:center'>{text}</p>"
centered_text2 = f"<p style='text-align:center'>{text2}</p>"
st.divider()
st.markdown(centered_text, unsafe_allow_html=True)
st.markdown(centered_text2, unsafe_allow_html=True)
