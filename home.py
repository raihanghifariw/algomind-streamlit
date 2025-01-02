# Import necessary libraries
import streamlit as st
import base64


def show():
    st.title("Sepsis Treatment Recommendation App")

    st.write(
        "Welcome to the Treatment Recommendations app powered by Deep Reinforcement Learning (DRL). This state-of-the-art application helps medical professionals determine optimal treatment strategies based on patient data, ensuring personalized and effective care."
    )

    # About YOLO (You Only Look Once)
    st.header("What is Sepsis?")
    st.write(
        "Sepsis is a life-threatening condition that occurs when the body's response to an infection causes widespread inflammation, leading to organ dysfunction or failure. It requires prompt medical intervention to prevent severe complications or death.\n"

        "Key facts about sepsis:"
    )
    st.markdown("- It can result from infections such as pneumonia, urinary tract infections, or bloodstream infections.\n"
                "- Symptoms may include fever, rapid heart rate, low blood pressure, confusion, and difficulty breathing.\n"
                "- Early diagnosis and timely, effective treatment are critical to improving patient outcomes.")

    # How Object Detection Works
    st.header("How Treatment Recommendations Work")
    st.write(
        "The app leverages DRL to analyze patient data and recommend the best course of treatment. By learning from historical cases and real-time inputs, the system identifies patterns and predicts the most effective interventions tailored to each patient. The DRL model iteratively improves its performance by optimizing reward functions tied to clinical outcomes."
    )

    # Example section for user interaction
    st.write("### Click to Know More About Sepsis")
    option = st.selectbox("Choose a topic to learn more", [
                          "Causes", "Prevention", "Symptoms", "Treatment"])

    if option == "Causes":
        st.write("### Causes of Sepsis")
        st.markdown("""
        - Infections in the lungs, urinary tract, abdomen, or bloodstream are common causes.
        - Pathogens such as bacteria, viruses, or fungi can trigger sepsis.
        - Chronic illnesses or weakened immune systems increase susceptibility.
        """)

    elif option == "Prevention":
        st.write("### Prevention of Sepsis")
        st.markdown("""
        - Stay up-to-date on vaccines.
        - Practice good hygiene, especially handwashing.
        - Treat infections promptly and appropriately.
        - Manage chronic conditions effectively.
        """)

    elif option == "Symptoms":
        st.write("### Symptoms of Sepsis")
        st.markdown("""
        - Fever or hypothermia
        - Elevated heart rate (tachycardia)
        - Increased respiratory rate (tachypnea)
        - Altered mental state
        - Low blood pressure in severe cases
        """)

    elif option == "Treatment":
        st.write("### Treatment for Sepsis")
        st.markdown("""
        - **Immediate Care**: Early recognition and treatment are vital.
        - **Antibiotics**: Administer broad-spectrum antibiotics as soon as possible.
        - **Fluids**: Intravenous fluids help stabilize blood pressure.
        - **Advanced Support**: In severe cases, oxygen therapy or organ support may be required.
        """)


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
        "For more detailed information and resources about sepsis, you can visit the following websites:"
    )
    st.markdown(
        "- [Sepsis Alliance](https://www.sepsis.org/)\n"
        "- [World Health Organization (WHO) - Sepsis](https://www.who.int/news-room/questions-and-answers/item/sepsis)\n"
        "- [Centers for Disease Control and Prevention (CDC) - Sepsis](https://www.cdc.gov/sepsis/index.html)"
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
