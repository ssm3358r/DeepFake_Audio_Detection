import os
import tempfile

import joblib
import librosa
import numpy as np
import streamlit as st

from audio_features import extract_features_for_mode, resolve_feature_mode_from_dim

TARGET_SR = 16000
FRAME_SECONDS = 3.0
HOP_SECONDS = 1.5
DEFAULT_FAKE_THRESHOLD = 0.70
MODEL_CANDIDATES = {
    "XGBoost (New)": ("deepfake_model_xgb.pkl", "scaler_xgb.pkl"),
    "RandomForest (Old)": ("deepfake_model1.pkl", "scaler.pkl"),
}


@st.cache_resource
def load_artifacts(model_path: str, scaler_path: str):
    model_obj = joblib.load(model_path)
    scaler_obj = joblib.load(scaler_path)
    feature_dim = getattr(scaler_obj, "n_features_in_", None)
    if feature_dim is None:
        feature_dim = getattr(model_obj, "n_features_in_", None)
    if feature_dim is None:
        raise ValueError("Unable to detect feature dimension from model/scaler.")
    feature_mode = resolve_feature_mode_from_dim(int(feature_dim))
    return model_obj, scaler_obj, feature_mode


def build_feature_windows(file_path: str, feature_mode: str) -> np.ndarray:
    audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)

    audio, _ = librosa.effects.trim(audio, top_db=25)
    if audio.size == 0:
        raise ValueError("Audio is silent after trimming.")
    audio = librosa.util.normalize(audio)

    frame_len = int(FRAME_SECONDS * sr)
    hop_len = int(HOP_SECONDS * sr)

    if audio.shape[0] < frame_len:
        return extract_features_for_mode(audio, sr, feature_mode).reshape(1, -1)

    windows = []
    for start in range(0, audio.shape[0] - frame_len + 1, hop_len):
        segment = audio[start : start + frame_len]
        windows.append(extract_features_for_mode(segment, sr, feature_mode))

    if not windows:
        windows.append(extract_features_for_mode(audio, sr, feature_mode))

    return np.asarray(windows)


def predict_audio(file_path: str, model, scaler, feature_mode: str, fake_threshold: float):
    feature_windows = build_feature_windows(file_path, feature_mode)
    scaled = scaler.transform(feature_windows)

    if hasattr(model, "predict_proba"):
        fake_scores = model.predict_proba(scaled)[:, 1]
        mean_fake_score = float(np.mean(fake_scores))
        pred_label = int(mean_fake_score >= fake_threshold)
        confidence = abs(mean_fake_score - 0.5) * 2
    else:
        votes = model.predict(scaled)
        pred_label = int(np.mean(votes) >= 0.5)
        mean_fake_score = float(np.mean(votes))
        confidence = abs(mean_fake_score - 0.5) * 2

    return pred_label, mean_fake_score, confidence, feature_windows.shape[0]


st.set_page_config(page_title="Deepfake Audio Detector", layout="centered")
st.title("Deepfake Audio Detection")
st.write("Upload an audio file or record voice to estimate whether it is real or fake.")

available_models = {
    name: paths
    for name, paths in MODEL_CANDIDATES.items()
    if os.path.exists(paths[0]) and os.path.exists(paths[1])
}

if not available_models:
    st.error("No model artifacts found. Train a model first.")
    st.stop()

model_choice = st.selectbox(
    "Choose model",
    list(available_models.keys()),
    index=0,
)

selected_model_path, selected_scaler_path = available_models[model_choice]
model, scaler, feature_mode = load_artifacts(selected_model_path, selected_scaler_path)
st.caption(f"Active model: {selected_model_path} | feature set: {feature_mode}")
fake_threshold = st.slider(
    "Fake decision threshold (higher = fewer false fake alerts)",
    min_value=0.50,
    max_value=0.95,
    value=DEFAULT_FAKE_THRESHOLD,
    step=0.01,
)

uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "flac", "m4a"])
recorded_audio = None
if hasattr(st, "audio_input"):
    st.write("Or record your voice directly:")
    recorded_audio = st.audio_input("Record voice")
else:
    st.info("Your Streamlit version does not support microphone recording yet.")

input_audio = recorded_audio if recorded_audio is not None else uploaded_file

if input_audio is not None:
    temp_path = None
    try:
        st.write("Audio preview:")
        st.audio(input_audio, format="audio/wav")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(input_audio.read())
            temp_path = tmp_file.name

        prediction, fake_score, confidence, windows_used = predict_audio(
            temp_path, model, scaler, feature_mode, fake_threshold
        )

        st.markdown("## Result")
        if prediction == 0:
            st.success("Prediction: Real Voice")
        else:
            st.error("Prediction: Fake Voice")

        st.write(f"Fake probability: `{fake_score:.3f}`")
        st.write(f"Threshold used: `{fake_threshold:.2f}`")
        st.write(f"Confidence: `{confidence:.3f}`")
        st.write(f"Windows analyzed: `{windows_used}`")

    except Exception as exc:
        st.warning(f"Error processing this file: {exc}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
