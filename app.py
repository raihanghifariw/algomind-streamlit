import streamlit as st
from sidebar import sidebar_menu
import home
import demo
import data
import visualisasi
import aboutus
# Menggunakan sidebar untuk memilih menu
menu = sidebar_menu()

# Memanggil konten berdasarkan menu yang dipilih
if menu == "Home":
    home.show()
elif menu == "Demo":
    demo.show()
elif menu == "Data":
    data.show()
elif menu == "Visualisasi":
    visualisasi.show()
elif menu == "About Us":
    aboutus.show()
