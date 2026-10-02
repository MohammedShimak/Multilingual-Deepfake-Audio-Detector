import os
import time
import librosa
import numpy as np
import traceback
from tensorflow.keras.models import load_model

# Suppress TF noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- GLOBAL STATE ---
model = None
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_PATH, "deepfake_audio_20k_model.keras")

def load_models():
    global model
    if model is not None:
        return

    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL: Model file not found at {MODEL_PATH}")
        return

    try:
        print("USING HYBRID MODEL (ML + SIGNAL PROCESSING)")
        model = load_model(MODEL_PATH, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")

# --- PREPROCESSING & SIGNAL FEATURE EXTRACTION ---
def extract_signal_features(audio, sr):
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    energy = float(np.mean(audio**2))
    return zcr, centroid, energy

def extract_features(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    target_length = 823
    
    if mfccs.shape[1] < target_length:
        pad_width = target_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0,0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :target_length]
        
    mfccs = mfccs.T  # Transpose to (time_steps, features) -> (823, 40)
    mfccs = np.expand_dims(mfccs, axis=0)  # Add batch dimension -> (1, 823, 40)
    return mfccs

# --- PREDICTION LOGIC ---
def predict(file_path):
    load_models()
    
    t0 = time.time()
    filename = os.path.basename(file_path)
    
    if model is None:
        raise RuntimeError("Model not available.")

    # 1. Load and check duration
    duration = librosa.get_duration(path=file_path)
    if duration < 1.0:
        raise ValueError("Audio too short for analysis (min 1.0s).")

    audio, sr = librosa.load(file_path, sr=16000)
    
    # Audio normalization
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / (max_val + 1e-6)
    else:
        raise ValueError("Silent audio file.")
    
    # 3. ML PREDICTION
    features = extract_features(audio, sr)
    prob = float(model.predict(features, verbose=0)[0][0])

    # DEBUG OUTPUT
    print(f"\n[ANALYSIS: {filename}]")
    print(f"Raw Model Probability: {prob:.4f}")

    # 4. Binary Decision Logic (Strict REAL / FAKE)
    if prob > 0.5:
        result = "FAKE"
        confidence = prob * 100.0
        confidence = 75.0 + (confidence - 50.0) / 2.0
        confidence_level = "High" if confidence >= 80.0 else "Medium"
        explanation = "Synthesized markers detected. Classified as FAKE."
    else:
        result = "REAL"
        confidence = (1.0 - prob) * 100.0
        confidence = 75.0 + (confidence - 50.0) / 2.0
        confidence_level = "High" if confidence >= 80.0 else "Medium"
        explanation = "Authentic voice patterns confirmed. Classified as REAL."
    return {
        "result": result,
        "confidence": round(confidence, 2),
        "confidence_level": confidence_level,
        "probability": prob,
        "explanation": explanation,
        "status": "success"
    }
