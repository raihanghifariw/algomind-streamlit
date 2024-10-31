# Import necessary libraries
import streamlit as st
import base64


def show():
    st.title("Algomind Object Detection")

    st.write(
        "Welcome to the Object Detection app powered by YOLO. This application uses a pre-trained YOLO model "
        "to detect objects within uploaded images, providing insights into object locations and classifications."
    )

    # About YOLO (You Only Look Once)
    st.header("What is YOLO?")
    st.write(
        "YOLO, short for 'You Only Look Once,' is a deep learning algorithm known for real-time object detection. "
        "It can rapidly identify and locate multiple objects within an image, making it suitable for applications "
        "where speed and accuracy are essential."
    )

    # How Object Detection Works
    st.header("How Object Detection Works")
    st.write(
        "Object detection involves identifying and classifying objects in an image. The YOLO algorithm scans the "
        "entire image in one pass, dividing it into a grid and predicting bounding boxes and class probabilities "
        "for each section."
    )

    # Use Cases
    st.header("Use Cases")
    st.write(
        "Object detection with YOLO has diverse applications, including surveillance, autonomous vehicles, "
        "medical imaging, and retail analytics. By understanding object locations and types, businesses can "
        "make informed decisions in real time."
    )

    # Instructions
    st.header("Instructions")
    st.write(
        "To use this app, simply upload an image in JPG, JPEG, or PNG format. The app will process the image and "
        "display the detected objects with bounding boxes and labels. Please ensure that images are clear and contain "
        "objects in well-lit conditions for the best results."
    )

    # Additional resources or links
    st.header("Additional Resources")
    st.write(
        "For more information on YOLO and its applications, you can visit the following resources:"
    )
    st.markdown(
        "- [YOLO Official GitHub Repository](https://github.com/ultralytics/yolov5)\n"
        "- [YOLO Paper (Original Research)](https://arxiv.org/abs/1506.02640)\n"
        "- [Deep Learning for Computer Vision Resources](https://www.deeplearning.ai/)"
    )

    # Footer
    st.write(
        "This Object Detection app is developed using Streamlit and YOLO. For inquiries or feedback, please "
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
