import streamlit as st
import pickle

# Direktori file pickle
exportdir = 'D:/TrekAI_ICU/'

# Fungsi untuk memuat model dan data dari file pickle


@st.cache(allow_output_mutation=True)
def load_models():
    with open(exportdir + 'bestpol.pkl', 'rb') as file:
        modl = pickle.load(file)
        Qon = pickle.load(file)
        physpol = pickle.load(file)
        transitionr = pickle.load(file)
        transitionr2 = pickle.load(file)
        R = pickle.load(file)
        C = pickle.load(file)
        train = pickle.load(file)
        qldata3train = pickle.load(file)
        qldata3test = pickle.load(file)

    with open(exportdir + 'step_5_start.pkl', 'rb') as file:
        MIMICzs = pickle.load(file)
        actionbloc = pickle.load(file)
        reformat5 = pickle.load(file)
        recqvi = pickle.load(file)

    return {
        'modl': modl,
        'Qon': Qon,
        'physpol': physpol,
        'transitionr': transitionr,
        'transitionr2': transitionr2,
        'R': R,
        'C': C,
        'train': train,
        'qldata3train': qldata3train,
        'qldata3test': qldata3test,
        'MIMICzs': MIMICzs,
        'actionbloc': actionbloc,
        'reformat5': reformat5,
        'recqvi': recqvi
    }


# Load model dan policy
model_data = load_models()

# Membuat input untuk icustayid
st.title("Demo Rekomendasi Perawatan Pasien")
icustayid = st.text_input("Masukkan ICUSTAYID:")

# Fungsi dummy untuk menghasilkan rekomendasi, bisa diganti dengan logika rekomendasi yang Anda punya


def get_rekomendasi(icustayid, model_data):
    # Gunakan `icustayid` dan `model_data` untuk menghasilkan rekomendasi berdasarkan model
    # Contoh sederhana:
    rekomendasi = f"Rekomendasi perawatan untuk ICUSTAYID {icustayid} berdasarkan model."
    return rekomendasi


# Menampilkan rekomendasi jika icustayid telah dimasukkan
if icustayid:
    rekomendasi = get_rekomendasi(icustayid, model_data)
    st.write(rekomendasi)
