from io import BytesIO
from matplotlib.animation import HTMLWriter
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
from itertools import product
import torch
from matplotlib.animation import FuncAnimation
from WD3QNE_deepQnet import WD3QNE  
from WD3QNE_evaluate import do_eval, do_test

# def show():
#     # =================Load model for deployment==================

#     def load_model(model_path, state_dim, num_actions):
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"Model file not found: {model_path}")

#         # Initialize the model
#         model = WD3QNE(state_dim=state_dim, num_actions=num_actions)

#         # Load the saved state_dicts
#         checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
#         for q_net, state_dict in zip(model.Q_ensemble, checkpoint['Q_state_dicts']):
#             q_net.load_state_dict(state_dict)
#         for q_net, state_dict in zip(model.Q_target_ensemble, checkpoint['Q_target_state_dicts']):
#             q_net.load_state_dict(state_dict)

#         print(f"Model loaded successfully from {model_path}")
#         print(
#             f"Best mean agent Q from training: {checkpoint['best_mean_agent_q']}")

#         return model  # Fungsi untuk menghasilkan data sinusoidal seperti sinyal heart rate

#     # Set file paths
#     DATA_PATH = "main/data/"
#     AGENT_ACTIONS_FILE = os.path.join(
#         DATA_PATH, "WD3QNE-algorithm/agent_actionsb.npy")
#     PHYS_ACTIONS_FILE = os.path.join(
#         DATA_PATH, "WD3QNE-algorithm/phys_actionsb.npy")
#     CSV_FILE = os.path.join(DATA_PATH, "rl_test_data_final_cont_new.csv")
#     MODEL_FILE = os.path.join(DATA_PATH, "WD3QNE-algorithm/best_model_ensemble.pt")

#     # Load data and predictions
#     data = pd.read_csv(CSV_FILE)
#     agent_actions = np.load(AGENT_ACTIONS_FILE)
#     phys_actions = np.load(PHYS_ACTIONS_FILE)

#     # Load model
#     state_dim = 37  # Adjust to match model configuration
#     n_actions = 25
#     model = load_model(MODEL_FILE, state_dim, n_actions)

#     # Streamlit Page Configuration
#     st.title("WD3QNE Action Prediction Demo")
#     st.divider()
#     st.write("This page demonstrates the WD3QNE model for predicting actions based on ICU patient data.")

#     # Select Patient ID
#     options = data['icustayid'].unique()
#     selected_option = st.selectbox("Select a Patient ID:", options)
#     save_indexes = data[data['icustayid'] == selected_option].index.tolist()
#     st.write(f"You selected: {selected_option}")
#     st.divider()

#     # Display Patient's Condition
#     st.subheader("Selected Patient's Condition in ICU")
#     selected_param = ["SOFA", "GCS", "SIRS", "Temp_C", "SysBP", "DiaBP", "HR"]
#     show_table = data.loc[save_indexes, selected_param].reset_index(drop=True)

#     plt.figure(figsize=(10, 6))
#     for param in selected_param:
#         plt.plot(show_table[param], label=param)
        

#     # plt.xlabel('4-Hourly Timestamps')
#     # plt.ylabel('Normalized Values')
#     # plt.title('Parameters Over Time')
#     # plt.legend()
#     # plt.grid(True)
#     # st.pyplot(plt)
#     # plt.close()

#     # Placeholder untuk animasi
#     placeholder = st.empty()

#     # Animasi Grafik
#     for t in range(1, len(show_table) + 1):
#         with placeholder.container():
#             plt.figure(figsize=(10, 6))
#             for param in selected_param:
#                 plt.plot(show_table[param][:t], label=param)

#             plt.xlabel('4-Hourly Timestamps')
#             plt.ylabel('Normalized Values')
#             plt.title('Parameters Over Time')
#             plt.legend()
#             plt.grid(True)
#             st.pyplot(plt)
#             plt.close()

#         # Waktu tunda untuk animasi
#         time.sleep(0.5)


#     # Predict Button
#     st.subheader("Make Treatment Recommendation With AI")
#     if st.button("Predict"):
#         st.subheader("Dose Intensity Recommendation for the Patient")
#         phys_act, ai_act = [], []
#         doses_intensity = ["Zero", "Low", "Medium", "High", "Very High"]
#         pair_of_act = list(product(doses_intensity, repeat=2))

#         for i in save_indexes:
#             phys_act.append(pair_of_act[phys_actions[i]])
#             ai_act.append(pair_of_act[agent_actions[i]])

#             temp = {
#                 'IV Fluid': {'Physician': phys_act[-1][0], 'AI': ai_act[-1][0]},
#                 'Vasopressor': {'Physician': phys_act[-1][1], 'AI': ai_act[-1][1]}
#             }
#             df = pd.DataFrame(temp)
#             st.table(df)
#             st.divider()

#         for i in save_indexes:
#             phys_act.append(pair_of_act[phys_actions[i]])
#             ai_act.append(pair_of_act[agent_actions[i]])

#         # Differences in Actions
#         st.subheader("Differences in actions taken between Physician and AI")
#         placeholder1 = st.empty()
#         placeholder2 = st.empty()

#         # Labels
#         y_labels = ["Zero", "Low", "Medium", "High", "Very High"]


#         # Animasi Grafik
#         for t in range(1, len(phys_act) + 1):
#             with placeholder1.container():
#                 y_values_phys = [y_labels.index(act[0]) for act in phys_act[:t]]
#                 y_values_ai = [y_labels.index(act[0]) for act in ai_act[:t]]

#                 # Plot untuk IV Fluid Dose Intensity
#                 fig, ax = plt.subplots()
#                 ax.plot(range(len(y_values_phys)), y_values_phys,
#                         label='Physician', marker='o', linestyle='-')
#                 ax.plot(range(len(y_values_ai)), y_values_ai,
#                         label='AI', marker='o', linestyle='-')
#                 ax.set_xlabel('4-Hourly Timestamps')
#                 ax.set_ylabel('Intensity')
#                 ax.set_title('IV Fluid Dose Intensity')
#                 ax.set_yticks(range(len(y_labels)))
#                 ax.set_yticklabels(y_labels)
#                 ax.legend()
#                 st.pyplot(fig)
#                 plt.close(fig)

#             with placeholder2.container():
#                 y_values_phys = [y_labels.index(act[1]) for act in phys_act[:t]]
#                 y_values_ai = [y_labels.index(act[1]) for act in ai_act[:t]]

#                 # Plot untuk Vasopressor Dose Intensity
#                 fig, ax = plt.subplots()
#                 ax.plot(range(len(y_values_phys)), y_values_phys,
#                         label='Physician', marker='o', linestyle='-')
#                 ax.plot(range(len(y_values_ai)), y_values_ai,
#                         label='AI', marker='o', linestyle='-')
#                 ax.set_xlabel('4-Hourly Timestamps')
#                 ax.set_ylabel('Intensity')
#                 ax.set_title('Vasopressor Dose Intensity')
#                 ax.set_yticks(range(len(y_labels)))
#                 ax.set_yticklabels(y_labels)
#                 ax.legend()
#                 st.pyplot(fig)
#                 plt.close(fig)

#             # Waktu tunda untuk animasi
#             time.sleep(0.5)
def show():
    # =================Load model for deployment==================

    def load_model(model_path, state_dim, num_actions):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Initialize the model
        model = WD3QNE(state_dim=state_dim, num_actions=num_actions)

        # Load the saved state_dicts
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        for q_net, state_dict in zip(model.Q_ensemble, checkpoint['Q_state_dicts']):
            q_net.load_state_dict(state_dict)
        for q_net, state_dict in zip(model.Q_target_ensemble, checkpoint['Q_target_state_dicts']):
            q_net.load_state_dict(state_dict)

        print(f"Model loaded successfully from {model_path}")
        print(
            f"Best mean agent Q from training: {checkpoint['best_mean_agent_q']}")

        return model

    # Set file paths
    DATA_PATH = "main/data/"
    AGENT_ACTIONS_FILE = os.path.join(
        DATA_PATH, "WD3QNE-algorithm/agent_actionsb.npy")
    PHYS_ACTIONS_FILE = os.path.join(
        DATA_PATH, "WD3QNE-algorithm/phys_actionsb.npy")
    CSV_FILE = os.path.join(DATA_PATH, "rl_test_data_final_cont_new.csv")
    MODEL_FILE = os.path.join(
        DATA_PATH, "WD3QNE-algorithm/best_model_ensemble.pt")

    # Load data and predictions
    data = pd.read_csv(CSV_FILE)
    agent_actions = np.load(AGENT_ACTIONS_FILE)
    phys_actions = np.load(PHYS_ACTIONS_FILE)

    # Load model
    state_dim = 37  # Adjust to match model configuration
    n_actions = 25
    model = load_model(MODEL_FILE, state_dim, n_actions)

    # Streamlit Page Configuration
    st.title("WD3QNE Action Prediction Demo")
    st.divider()
    st.write(
        "This page demonstrates the WD3QNE model for predicting actions based on ICU patient data.")

    # Select Patient ID
    options = data['icustayid'].unique()
    selected_option = st.selectbox("Select a Patient ID:", options)
    save_indexes = data[data['icustayid'] == selected_option].index.tolist()
    st.write(f"You selected: {selected_option}")
    st.divider()

    # Display Patient's Condition
    st.subheader("Selected Patient's Condition in ICU")
    selected_param = ["SOFA", "GCS", "SIRS", "Temp_C", "SysBP", "DiaBP", "HR"]
    show_table = data.loc[save_indexes, selected_param].reset_index(drop=True)

    # Placeholder untuk animasi
    placeholder = st.empty()

    # Animasi Grafik Parameter Pasien
    for t in range(1, len(show_table) + 1):
        with placeholder.container():
            plt.figure(figsize=(10, 6))
            for param in selected_param:
                plt.plot(show_table[param][:t], label=param)

            plt.xlabel('4-Hourly Timestamps')
            plt.ylabel('Normalized Values')
            plt.title('Parameters Over Time')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)
            plt.close()

        # Waktu tunda untuk animasi
        time.sleep(0.5)

    # Treatment Recommendation
    st.subheader("Dose Intensity Recommendation for the Patient")
    phys_act, ai_act = [], []
    doses_intensity = ["Zero", "Low", "Medium", "High", "Very High"]
    pair_of_act = list(product(doses_intensity, repeat=2))

    for i in save_indexes:
        phys_act.append(pair_of_act[phys_actions[i]])
        ai_act.append(pair_of_act[agent_actions[i]])

        temp = {
            'IV Fluid': {'Physician': phys_act[-1][0], 'AI': ai_act[-1][0]},
            'Vasopressor': {'Physician': phys_act[-1][1], 'AI': ai_act[-1][1]}
        }
        df = pd.DataFrame(temp)
        st.table(df)
        st.divider()

    # Differences in Actions
    st.subheader("Differences in actions taken between Physician and AI")
    placeholder1 = st.empty()
    placeholder2 = st.empty()

    # Labels
    y_labels = ["Zero", "Low", "Medium", "High", "Very High"]

    # Animasi Grafik
    for t in range(1, len(phys_act) + 1):
        with placeholder1.container():
            y_values_phys = [y_labels.index(act[0]) for act in phys_act[:t]]
            y_values_ai = [y_labels.index(act[0]) for act in ai_act[:t]]

            # Plot untuk IV Fluid Dose Intensity
            fig, ax = plt.subplots()
            ax.plot(range(len(y_values_phys)), y_values_phys,
                    label='Physician', marker='o', linestyle='-')
            ax.plot(range(len(y_values_ai)), y_values_ai,
                    label='AI', marker='o', linestyle='-')
            ax.set_xlabel('4-Hourly Timestamps')
            ax.set_ylabel('Intensity')
            ax.set_title('IV Fluid Dose Intensity')
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels)
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        with placeholder2.container():
            y_values_phys = [y_labels.index(act[1]) for act in phys_act[:t]]
            y_values_ai = [y_labels.index(act[1]) for act in ai_act[:t]]

            # Plot untuk Vasopressor Dose Intensity
            fig, ax = plt.subplots()
            ax.plot(range(len(y_values_phys)), y_values_phys,
                    label='Physician', marker='o', linestyle='-')
            ax.plot(range(len(y_values_ai)), y_values_ai,
                    label='AI', marker='o', linestyle='-')
            ax.set_xlabel('4-Hourly Timestamps')
            ax.set_ylabel('Intensity')
            ax.set_title('Vasopressor Dose Intensity')
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels)
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        # Waktu tunda untuk animasi
        time.sleep(0.5)
