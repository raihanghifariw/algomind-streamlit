
# Import necessary libraries
import streamlit as st
import base64


def show():
    st.title("Algomind")

    st.write(
        "Welcome to the Sepsis Treatment Recommendation app. This application "
        "provides recommendations for sepsis treatment based on several clinical parameters."
    )

    # What is sepsis?
    st.header("What is Sepsis?")
    st.write(
        "Sepsis is a severe illness caused by the body's response to an infection. It can lead to organ "
        "failure and, if not treated promptly, can be fatal. Early detection and intervention are crucial "
        "for successful treatment."
    )

    # Symptoms of sepsis
    st.header("Symptoms of Sepsis")
    st.write(
        "Sepsis symptoms can vary, but common signs include fever, elevated heart rate, rapid breathing, "
        "confusion, and extreme discomfort. If you suspect sepsis, seek medical attention immediately."
    )

    # Risk factors
    st.header("Risk Factors")
    st.write(
        "Certain factors increase the risk of developing sepsis, including age, weakened immune system, "
        "chronic medical conditions, and recent surgery or invasive procedures."
    )

    # Prevention and treatment
    st.header("Prevention and Treatment")
    st.write(
        "Preventing infections, practicing good hygiene, and seeking prompt medical attention for infections "
        "can help prevent sepsis. Treatment involves antibiotics, supportive care, and addressing the underlying cause."
    )

    # Additional resources or links
    st.header("Additional Resources")
    st.write(
        "For more detailed information and resources about sepsis, you can visit the following websites:"
    )
    st.markdown(
        "- [Sepsis Alliance](https://www.sepsis.org/)\n"
        "- [World Health Organization (WHO) - Sepsis](https://www.who.int/news-room/questions-and-answers/item/sepsis)\n"
        "- [Centers for Disease Control and Prevention (CDC) - Sepsis](https://www.cdc.gov/sepsis/index.html)"
    )

    # Footer
    st.write(
        "This Sepsis Treatment Recommendation app is created using Streamlit. If you have any questions or feedback, please "
        "contact HFR Teams at YARSI University."
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
