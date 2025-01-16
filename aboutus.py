from PIL import Image
import streamlit as st
import json
import time

# Initialize member index in session_state
if 'member_index' not in st.session_state:
    st.session_state['member_index'] = 0


def show():
    st.title("About Us")
    st.write(
        "We are the Algomind team of three dedicated individuals working on the YOLO Object Detection project. "
        "Our mission is to provide an efficient and user-friendly application for real-time object detection."
    )

    # Team Header
    st.header("Meet Our Team")

    # List of team members
    team_members = [
        {
            "name": "Raihan Ghifari Winata",
            "image": "main/assets/gipari.jpeg",
            "description": "Ghifari is a computer vision expert with a passion for deep learning. Connect with Ghifari on [LinkedIn](https://www.linkedin.com/in/raihan-ghifari-553a1a26a/)"
        },
        {
            "name": "Karina Defitrah Nurul Jinan",
            "image": "main/assets/karina.jpg",
            "description": "Karina is a software engineer specializing in building scalable applications. Connect with Karina on [LinkedIn](https://www.linkedin.com/in/karina-defitrah-nurul-jinan-0520b1303/)"
        },
        {
            "name": "Yahya Alghazali Mushlih",
            "image": "main/assets/yahya.jpeg",
            "description": "Yahya is a data scientist focused on leveraging data for impactful solutions. Connect with Yahya on [LinkedIn](https://www.linkedin.com/in/yahya-alghazali-mushlih-40280528b/)"
        },
    ]

    # Get the current team member based on the index in session_state
    current_member = team_members[st.session_state['member_index']]

    # Create columns for image and description
    # col1, col2 = st.columns([1, 2])

    # with col1:
    #     # Open and display the image in the first column
    #     image = Image.open(current_member['image'])
    #     st.image(image, caption=current_member['name'], use_column_width=True)

    st.subheader(current_member['name'])
    st.write(current_member['description'])

    # Navigation buttons for Next and Previous
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Previous"):
            st.session_state['member_index'] = (
                st.session_state['member_index'] - 1) % len(team_members)
    with col2:
        # Open and display the image in the first column
        image = Image.open(current_member['image'])
        st.image(image, caption=current_member['name'], use_column_width=True)
    with col3:
        if st.button("Next"):
            st.session_state['member_index'] = (
                st.session_state['member_index'] + 1) % len(team_members)

    # Footer
    st.write(
        "Thank you for visiting our page. We are excited to share our work with you!"
    )
    st.write(
        "If you have any inquiries or would like to get in touch with our team, please email us at "
        "[algominang@gmail.com]. We appreciate your interest in our mission."
    )

    text = "©2024 Algomind Company. All rights reserved. The content on this website is protected by copyright law."
    text2 = "For permission requests, please contact us."

    # Using HTML tags for text alignment
    centered_text = f"<p style='text-align:center'>{text}</p>"
    centered_text2 = f"<p style='text-align:center'>{text2}</p>"
    st.divider()
    # Displaying centered text using st.markdown
    st.markdown(centered_text, unsafe_allow_html=True)
    st.markdown(centered_text2, unsafe_allow_html=True)
