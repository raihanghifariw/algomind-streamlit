import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st


def show():
    # Import necessary libraries
    # Page title
    st.title("MIMIC-III Dataset Information")


    # Introduction
    st.write(
        "The MIMIC-III (Medical Information Mart for Intensive Care III) dataset is a widely used "
        "publicly available dataset that contains de-identified health data for patients admitted to "
        "the intensive care unit (ICU). This page provides an overview of the MIMIC-III dataset."
    )

    # About MIMIC-III
    st.header("About MIMIC-III")
    st.write(
        "MIMIC-III is a database developed by the Massachusetts Institute of Technology (MIT) "
        "that includes clinical data from patients admitted to the Beth Israel Deaconess Medical Center in Boston, USA. "
        "The dataset covers a wide range of information, including demographics, vital signs, laboratory test results, "
        "medication orders, and more."
    )

    # Accessing the dataset
    st.header("Accessing the Dataset")
    st.write(
        "Researchers and developers can access the MIMIC-III dataset by completing the necessary "
        "training and obtaining access through the PhysioNet website. More information on accessing the dataset can be found "
        "on the [MIMIC-III website](https://mimic.physionet.org/gettingstarted/access/)."
    )

    # Dataset Features
    st.header("Key Features of the MIMIC-III Dataset")
    st.write(
        "The MIMIC-III dataset includes a variety of features, such as:\n"
        "- Patient demographics\n"
        "- Vital signs (e.g., heart rate, blood pressure)\n"
        "- Laboratory test results\n"
        "- Medication orders and administrations\n"
        "- Diagnoses and procedures\n"
        "- Clinical notes and reports\n"
        "- And more."
    )

    # Data Usage and Citation
    st.header("Data Usage and Citation")
    st.write(
        "Users of the MIMIC-III dataset are expected to adhere to the data use agreement and cite the original "
        "publications associated with the dataset. Please refer to the [MIMIC-III citation page](https://mimic.physionet.org/about/cite/) "
        "for detailed information on citing the dataset in your work."
    )

    # Footer
    st.write(
        "For more detailed information and documentation about the MIMIC-III dataset, please refer to the official "
        "[MIMIC-III website](https://mimic.physionet.org/). If you have any questions or inquiries, please contact the "
        "MIMIC-III research team through their [contact page](https://mimic.physionet.org/about/team/)."
    )

    # Footer
    st.write(
        "This Sepsis Treatment Recommendation app is created using Streamlit. For inquiries or feedback, please "
        "contact the Algomind team at YARSI University."
    )

    text = "©2024 Algomind Company. All rights reserved. The content on this website is protected by copyright law."
    text2 = "For permission requests, please contact algomind@gmail.com."

    # Using HTML tags for text alignment
    centered_text = f"<p style='text-align:center'>{text}</p>"
    centered_text2 = f"<p style='text-align:center'>{text2}</p>"
    st.divider()
    # Displaying centered text using st.markdown
    st.markdown(centered_text, unsafe_allow_html=True)
    st.markdown(centered_text2, unsafe_allow_html=True)
