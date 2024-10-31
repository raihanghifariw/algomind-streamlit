import streamlit as st

# Buat main menu dropdown di sidebar
menu = st.sidebar.selectbox(
    "Pilih Menu:",
    ("Beranda", "Data", "Visualisasi", "Tentang")
)

# Menampilkan konten berdasarkan pilihan menu
if menu == "Beranda":
    st.title("Selamat Datang di Halaman Beranda")
    st.write("Ini adalah halaman utama aplikasi.")
elif menu == "Data":
    st.title("Data")
    st.write("Halaman ini untuk menampilkan data.")
elif menu == "Visualisasi":
    st.title("Visualisasi")
    st.write("Halaman ini untuk visualisasi data.")
elif menu == "Tentang":
    st.title("Tentang")
    st.write("Ini adalah aplikasi yang dibuat dengan Streamlit.")
