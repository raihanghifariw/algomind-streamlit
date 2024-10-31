import streamlit as st


def sidebar_menu():
    menu = st.sidebar.selectbox(
        "Pilih Menu:",
        ("Home", "Data","Visualisasi", "Demo", "About Us")
    )
    return menu
