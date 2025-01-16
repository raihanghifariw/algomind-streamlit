import streamlit as st


def sidebar_menu():
    menu = st.sidebar.selectbox(
        "Pilih Menu:",
        ("Home", "Data", "Demo", "About Us")
    )
    return menu
