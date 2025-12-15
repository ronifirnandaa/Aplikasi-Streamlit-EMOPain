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
# Load data referensi (UNTUK GENERATOR)
# ---------------------------
@st.cache_data
def load_reference_data():
    df = pd.read_csv("manual_testing_data.csv")
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y

X_ref, y_ref = load_reference_data()

LABEL_NAME = {
    0: "Sehat",
    1: "Nyeri rendah",
    2: "Nyeri tinggi"
}

# ---------------------------
# Generator berbasis DATA ASLI
# ---------------------------
def generate_from_real_data(label, noise_scale=0.02):
    idx = np.random.choice(y_ref[y_ref == label].index)
    base_sample = X_ref.loc[idx].values.copy()

    noise = np.random.normal(
        loc=0,
        scale=noise_scale * np.std(base_sample),
        size=base_sample.shape
    )

    generated = base_sample + noise
    generated = np.clip(generated, 0, 1)

    return generated.reshape(30, 200)

# ---------------------------
# Main Title
# ---------------------------
st.title("Prediksi Tingkat Nyeri dengan EMOPain (Random Forest)")
st.write("Pilih metode input data di sidebar.")

# ---------------------------
# Sidebar: Info Model
# ---------------------------
st.sidebar.title("Evaluasi Model Random Forest")

accuracy = 0.8628
st.sidebar.metric("Akurasi", f"{accuracy*100:.2f}%")

st.sidebar.markdown("---")

input_type = st.sidebar.radio(
    "Metode input data:",
    ["Upload CSV", "Input Manual 30 Channel"]
)

# ---------------------------
# INPUT MANUAL + GENERATOR
# ---------------------------
if input_type == "Input Manual 30 Channel":

    if "channel_data" not in st.session_state:
        st.session_state.channel_data = [[0.5]*200 for _ in range(30)]

    left, right = st.columns([3, 1])

    # ===== GENERATOR =====
    with right:
        st.subheader("Generate Data")

        gen_label = st.selectbox(
            "Label sumber data",
            [0, 1, 2],
            format_func=lambda x: f"{x} – {LABEL_NAME[x]}"
        )

        noise_level = st.slider(
            "Noise level",
            0.0, 0.1, 0.02, 0.01
        )

        if st.button("Generate Data"):
            generated = generate_from_real_data(gen_label, noise_level)
            st.session_state.channel_data = generated.tolist()
            st.success("Data berhasil digenerate")

    # ===== INPUT MANUAL (5 KOLOM / BARIS) =====
    channel_data = []

    with left:
        st.subheader("Input Manual 30 Channel")

        with st.expander("Tampilkan Input Channel"):
            for row in range(6):
                cols = st.columns(5)
                for col in range(5):
                    ch_idx = row * 5 + col

                    with cols[col]:
                        text = ",".join(
                            f"{x:.4f}" for x in st.session_state.channel_data[ch_idx]
                        )
                        user_input = st.text_area(
                            f"Ch {ch_idx + 1}",
                            value=text,
                            height=70
                        )

                    try:
                        values = [float(x.strip()) for x in user_input.split(",")]
                        if len(values) < 200:
                            values += [0.5] * (200 - len(values))
                        elif len(values) > 200:
                            values = values[:200]
                    except:
                        values = [0.5] * 200

                    channel_data.append(values)

    st.session_state.channel_data = channel_data

    # ===== DATA UNTUK PREDIKSI =====
    data = pd.DataFrame(
        np.array(channel_data).flatten().reshape(1, -1)
    )

    # ===== PREVIEW =====
    st.subheader("Preview Data")
    st.dataframe(pd.DataFrame(channel_data).T, height=200)

    # ===== VISUALISASI =====
    st.subheader("Visualisasi 30 Channel")
    fig, axes = plt.subplots(6, 5, figsize=(15, 8))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.plot(channel_data[i])
        ax.set_title(f"Ch {i+1}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    st.pyplot(fig)

# =========================================================
# UPLOAD CSV (TIDAK DIUBAH)
# =========================================================
elif input_type == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV (1 row × 6000 fitur + label optional)",
        type=["csv"]
    )

    if uploaded_file is not None:
        data_raw = pd.read_csv(uploaded_file)
        st.subheader("Data Sampel")
        st.dataframe(data_raw.head())

        if 'label' in data_raw.columns:
            data = data_raw.drop(columns=['label'])
            st.write("Kolom 'label' dihapus dari input fitur.")
        else:
            data = data_raw

        st.subheader("Preview Data (Ringkas)")
        preview_df = pd.DataFrame(
            data.values.reshape(30, 200).T,
            columns=[f"Ch{i+1}" for i in range(30)]
        )
        st.dataframe(preview_df, height=200)

        st.subheader("Visualisasi Semua 30 Channel")
        fig, axes = plt.subplots(6, 5, figsize=(15, 8), sharex=True)
        axes = axes.flatten()

        channel_values = data.values.reshape(30, 200)
        for i, ax in enumerate(axes):
            ax.plot(channel_values[i])
            ax.set_title(f"Ch {i+1}")
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        st.pyplot(fig)
    else:
        data = None


# ---------------------------
# PREDIKSI
# ---------------------------
st.markdown("---")
if st.button("Prediksi") and data is not None:
    X_input = data.values.astype(np.float64)
    pred = rf_model.predict(X_input)[0]

    st.success(f"Hasil Prediksi: {pred} → {LABEL_NAME[pred]}")
