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
import pickle as pkl
import pandas as pd
from itertools import product

with open('main/data/dqn_normal_actions_test(Best).p', 'rb') as file:
    agent_action_test = pkl.load(file)

with open('main/data/phys_actions_test.p', 'rb') as file:
    phys_action_test = pkl.load(file)

data = pd.read_csv('main/data/rl_test_data_final_cont.csv')

# Page title
st.title("Demo")
st.divider()
# Update this with the correct path to your logo
logo_path = "main/assets/sepvisor.jpeg"
st.sidebar.image(logo_path)
# Introduction
st.write(
    "This page demonstrates a simple prediction using a hypothetical machine learning model. "
)

options = data['icustayid'].drop_duplicates()
# Create a combo box using selectbox
selected_option = st.selectbox("Select an ICU Stay ID:", options)
save_indexes = []
i_index = 0
for i, row in data.iterrows():
    if data.loc[i, 'icustayid'] == selected_option:
        save_indexes.append(i)
# Display the selected option
st.write(f"You selected: {selected_option}")
st.divider()
# def show_table(selected_state):
st.subheader("Selected Patients Condition in ICU per-Stay")
selected_param = ["SOFA", "GCS", "SIRS", "Temp_C", "SysBP", "DiaBP", "HR"]
show_table = data[selected_param]
show_table = show_table.loc[save_indexes].reset_index(drop=True)
plt.figure(figsize=(10, 6))
plt.plot(show_table['SOFA'], label='SOFA')
plt.plot(show_table['GCS'], label='GCS')
plt.plot(show_table['SIRS'], label='SIRS')
plt.plot(show_table['Temp_C'], label='Temperature (C)')
plt.plot(show_table['SysBP'], label='SysBP')
plt.plot(show_table['DiaBP'], label='DiaBP')
plt.plot(show_table['HR'], label='Heart Rate')

# Customize the plot
plt.xlabel('4-Hourly Timestamps')
plt.ylabel('Normalized Values')
plt.title('Parameters Over Time')
plt.legend()
plt.grid(True)
st.pyplot(plt)
plt.close()
# Button to trigger prediction
st.subheader("Make Treatment Recommendation With AI")
if st.button("Predict"):
    st.subheader("Dose Intensity Recomendation for The Patients")

    phys_act = []
    ai_act = []
    vaso_phys = []
    iv_phys = []
    vaso_ai = []
    iv_ai = []
    doses_intensity = ["Zero", "Low", "Medium", "High", "Very High"]
    pair_of_act = list(product(doses_intensity, repeat=2))
    st.divider()
    # Your loop to fill the data
    for i in range(len(save_indexes)):
        st.text(f"Dose Intensity After {(i+1)*4} Hour")
        phys_act.append(pair_of_act[phys_action_test[save_indexes[i]]])
        ai_act.append(pair_of_act[agent_action_test[save_indexes[i]]])
        vaso_phys.append(phys_act[i][0])
        iv_phys.append(phys_act[i][1])
        vaso_ai.append(ai_act[i][0])
        iv_ai.append(ai_act[i][1])
        temp = {
            'IV Fluid': {'Physician': phys_act[i][0], 'AI': ai_act[i][0]},
            'Vasopressor': {'Physician': phys_act[i][1], 'AI': ai_act[i][1]}
        }
        df = pd.DataFrame(temp)
        st.table(df)
        st.divider()

    # Data
    # doses_intensity_set1 = ['High', 'Low', 'Zero', 'High', 'High', 'Low']
    # doses_intensity_set2 = ['Medium', 'High', 'Low', 'Medium', 'High', 'Medium']
    st.subheader("Differences in action taken between clinician and AI")
    col1, col2 = st.columns(2)


# Team member 1 in the first column
    with col1:
        y_labels = ["Zero", "Low", "Medium", "High", "Very High"]

        # Map doses_intensity to corresponding numerical values
        y_values_set1 = [y_labels.index(intensity) for intensity in iv_phys]
        y_values_set2 = [y_labels.index(intensity) for intensity in iv_ai]

        # Create a line plot for each set
        plt.plot(range(len(iv_phys)), y_values_set1,
                 label='Physician', marker='o', linestyle='-')
        plt.plot(range(len(iv_ai)), y_values_set2,
                 label='AI', marker='o', linestyle='-')

        # Customize the plot
        plt.xlabel('Index 4-Hourly Timestamps')
        plt.ylabel('Intensity')
        plt.title('Vasopressor Doses Intensity Line Plot')
        plt.yticks(range(len(y_labels)), y_labels)
        plt.legend()
        st.pyplot(plt)
        plt.close()

    # Team member 2 in the second column
    with col2:
        y_labels = ["Zero", "Low", "Medium", "High", "Very High"]

        # Map doses_intensity to corresponding numerical values
        y_values_set3 = [y_labels.index(intensity) for intensity in vaso_phys]
        y_values_set4 = [y_labels.index(intensity) for intensity in vaso_ai]

        # Create a line plot for each set
        plt.plot(range(len(vaso_phys)), y_values_set3,
                 label='Physician', marker='o', linestyle='-')
        plt.plot(range(len(vaso_ai)), y_values_set4,
                 label='AI', marker='o', linestyle='-')

        # Customize the plot
        plt.xlabel('Index 4-Hourly Timestamps')
        plt.ylabel('Intensity')
        plt.title('Vasopressor Doses Intensity Line Plot')
        plt.yticks(range(len(y_labels)), y_labels)
        plt.legend()
        st.pyplot(plt)

    # show_sofa = st.checkbox("SOFA Score")
    # show_gcs = st.checkbox("GCS")
    # Conditionally display text based on the checkbox value
    # if show_sofa:
    #    selected_param.append("SOFA")
    # else:
    #    selected_param.remove("SOFA")
    # st.write(selected_param)

    # Creating a DataFrame
    # df = pd.DataFrame(data)

    # Displaying the table in Streamlit
    # st.table(df)
    # st.write(f"{save_indexes}")

# Footer

text = "©2024 HFR Company. All rights reserved. The content on this website is protected by copyright law."
text2 = "For permission requests, please contact +6287870190448."

# Using HTML tags for text alignment
centered_text = f"<p style='text-align:center'>{text}</p>"
centered_text2 = f"<p style='text-align:center'>{text2}</p>"
st.divider()
# Displaying centered text using st.markdown
st.markdown(centered_text, unsafe_allow_html=True)
st.markdown(centered_text2, unsafe_allow_html=True)
