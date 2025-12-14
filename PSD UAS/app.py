import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------------------
# Load model Random Forest
# ---------------------------
rf_model = joblib.load("rf_clf.pkl")

# ---------------------------
# Sidebar: Info Model
# ---------------------------
st.sidebar.title("Evaluasi Model Random Forest")
st.sidebar.markdown("Ringkasan evaluasi model pada data test:")

accuracy = 0.8628
class_counts = {0: 735, 1: 136, 2: 81}
target_names = {0: "Sehat", 1: "Nyeri rendah", 2: "Nyeri tinggi"}

st.sidebar.metric("Akurasi", f"{accuracy*100:.2f}%")
st.sidebar.markdown("Distribusi kelas pada test set:")
for cls, count in class_counts.items():
    st.sidebar.write(f"{target_names[cls]}: {count} sampel")

st.sidebar.markdown("---")
st.sidebar.write("Gunakan menu utama untuk input data dan prediksi.")

# ---------------------------
# Main Title
# ---------------------------
st.title("Prediksi Tingkat Nyeri dengan EMOPain (Random Forest)")
st.write("Pilih metode input data di sidebar.")

# ---------------------------
# Pilihan input
# ---------------------------
input_type = st.sidebar.radio("Metode input data:", ["Upload CSV", "Input Manual 30 Channel"])

# ---------------------------
# Input Manual 30 Channel
# ---------------------------
if input_type == "Input Manual 30 Channel":
    st.subheader("Input Manual 30 Channel")
    st.write("Masukkan data tiap channel (1–200 titik). Channel kosong akan otomatis diisi default 0.5.")

    channel_data = []

    with st.expander("Klik untuk menampilkan input semua 30 channel"):
        for group_start in range(0, 30, 5):  # 5 channel per baris
            cols = st.columns(5)
            for i, ch in enumerate(range(group_start, min(group_start+5, 30))):
                with cols[i]:
                    user_input = st.text_area(
                        f"Ch {ch+1}",
                        value=",".join(["0.5"]*200),
                        height=80
                    )
                    try:
                        values = [float(x.strip()) for x in user_input.split(",")]
                        if len(values) < 200:
                            values += [0.5]*(200 - len(values))
                        elif len(values) > 200:
                            values = values[:200]
                    except:
                        st.error(f"Format salah di Channel {ch+1}! Menggunakan default 0.5")
                        values = [0.5]*200
                    channel_data.append(values)

    # Flatten data untuk prediksi
    data_array = np.array(channel_data).flatten().reshape(1, -1)
    data = pd.DataFrame(data_array)

    # Preview channel
    st.subheader("Preview Channel (Ringkas)")
    preview_df = pd.DataFrame(channel_data).T
    st.dataframe(preview_df, height=200)

    # Visualisasi channel
    st.subheader("Visualisasi Channel")
    n_rows = 6
    n_cols = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2*n_rows), sharex=True)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < len(channel_data):
            ax.plot(channel_data[i], color="#1f77b4")
            ax.set_title(f"Ch {i+1}")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    st.pyplot(fig)

# ---------------------------
# Upload CSV
# ---------------------------
elif input_type == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV (1 row × 6000 fitur + label optional)", type=["csv"])
    if uploaded_file is not None:
        data_raw = pd.read_csv(uploaded_file)
        st.subheader("Data Sampel")
        st.dataframe(data_raw.head())

        # Pisahkan label jika ada
        if 'label' in data_raw.columns:
            labels = data_raw['label']
            data = data_raw.drop(columns=['label'])
            st.write("Kolom 'label' dihapus dari input fitur untuk prediksi.")
        else:
            data = data_raw

        # Preview (seluruh channel)
        st.subheader("Preview Data (Ringkas)")
        preview_df = pd.DataFrame(data.values.reshape(30, 200).T, columns=[f"Ch{i+1}" for i in range(30)])
        st.dataframe(preview_df, height=200)

        # Visualisasi semua 30 channel
        st.subheader("Visualisasi Semua 30 Channel")
        fig, axes = plt.subplots(6, 5, figsize=(15, 8), sharex=True)
        axes = axes.flatten()
        channel_values = data.values.reshape(30, 200)
        for i, ax in enumerate(axes):
            if i < 30:
                ax.plot(channel_values[i], color="#1f77b4")
                ax.set_title(f"Ch {i+1}")
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        st.pyplot(fig)
    else:
        data = None
# ---------------------------
# Prediksi
# ---------------------------
if st.button("Prediksi") and data is not None:
    try:
        X_input = data.values.astype(np.float64)
        pred = rf_model.predict(X_input)
        label_mapping = {0: "Sehat", 1: "Nyeri rendah", 2: "Nyeri tinggi"}
        st.success(f"Hasil Prediksi: {pred[0]} → {label_mapping[pred[0]]}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
