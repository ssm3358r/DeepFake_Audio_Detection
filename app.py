import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile

# Load model + scaler
model = joblib.load("deepfake_model1.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.set_page_config(page_title="Deepfake Audio Detector", layout="centered")
st.title("🎧 Deepfake Audio Detection")
st.write("Upload an audio file to check whether it is real or fake.")

# Feature extraction
def extract_features(file):
    audio, sr = librosa.load(file, sr=16000)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    return np.hstack((mfcc_mean, mfcc_std))

# File upload
uploaded_file = st.file_uploader("📂 Upload Audio File", type=["wav", "mp3"])

# Prediction
if uploaded_file is not None:
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        # Extract features
        features = extract_features(temp_path)
        features = features.reshape(1, -1)

        # Apply scaler
        features = scaler.transform(features)

        # Predict
        prediction = model.predict(features)

        st.markdown("## 🔍 Result")

        if prediction[0] == 0:
            st.success("✅ Real Voice")
        else:
            st.error("❌ Fake Voice")

    except Exception as e:
        st.warning("⚠️ Error processing file")
