from PIL import Image
import streamlit as st


def show():
    # Page title
    st.title("About Us")
    # img_razi = Image.open("main/assets/razi.png")
    # img_hekal = Image.open("main/assets/hekal.png")
    # img_fadlu = Image.open("main/assets/fadlu.png")
    st.write(
        "Welcome to the 'About Us' page. Get to know the individuals behind the Sepvisor."
    )

    # Using st.columns to create three columns
    col1, col2, col3 = st.columns(3)

    # Team member 1 in the first column
    with col1:
        st.header("Raihan Ghifari Winata")
        # st.image(img_razi, caption="Razi", use_column_width=True)
        st.write(
            "Connect with Razi on [LinkedIn](https://www.linkedin.com/in/raihan-ghifari-553a1a26a/)")

    # Team member 2 in the second column
    with col2:
        st.header("Karina Defitrah Nurul Jinan")
        # st.image(img_fadlu, caption="Fadlu", use_column_width=True)
        st.write(
            "Connect with Fadlu on [LinkedIn](https://www.linkedin.com/in/karina-defitrah-nurul-jinan-0520b1303/)")

    # Team member 3 in the third column
    with col3:
        st.header("Yahya Alghazali Mushlih")
        # st.image(img_hekal, caption="Haikal", use_column_width=True)
        st.write(
            "Connect with Haikal on [LinkedIn](https://www.linkedin.com/in/yahya-alghazali-mushlih-40280528b/)")

    # Footer
    st.write(
        "If you have any inquiries or would like to get in touch with our team, please email us at "
        "[algominang@gmail.com]. We appreciate your interest in our mission."
    )

    text = "©2024 Algomins Company. All rights reserved. The content on this website is protected by copyright law."
    text2 = "For permission requests, please contact us."

    # Using HTML tags for text alignment
    centered_text = f"<p style='text-align:center'>{text}</p>"
    centered_text2 = f"<p style='text-align:center'>{text2}</p>"
    st.divider()
    # Displaying centered text using st.markdown
    st.markdown(centered_text, unsafe_allow_html=True)
    st.markdown(centered_text2, unsafe_allow_html=True)
