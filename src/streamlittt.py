import streamlit as st
import librosa
import numpy as np
import joblib

# Load the model
model = joblib.load("models/emotion_model_xgboost.pkl")

# Label mapping (update as per your model)
label_map = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised"
}

# Title
st.title("🎵 Emotion Recognition from Audio")

# Upload audio
uploaded_file = st.file_uploader("Upload an audio (.wav) file", type=["wav"])

# Feature extraction
def extract_features(audio_path, n_mfcc=40):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

# Predict emotion
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    # Extract features safely
    try:
        features = extract_features(uploaded_file)
        features = features.reshape(1, -1)

        # Predict
        predicted_class = model.predict(features)[0]
        emotion = label_map.get(predicted_class, "Unknown")
        st.success(f"Predicted Emotion: **{emotion}** 🎧")

    except Exception as e:
        st.error(f"Error processing audio: {e}")
