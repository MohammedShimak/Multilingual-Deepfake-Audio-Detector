import os
import librosa
import numpy as np
import xgboost as xgb
import pickle
import time
from scipy.signal import convolve2d

# Global models
cnn_weights = None
xgb_model = None

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_PATH = os.path.join(BASE_PATH, "cnn_weights.npz")
XGB_PATH = os.path.join(BASE_PATH, "xgb_model.pkl")

def load_models():
    global cnn_weights, xgb_model
    if cnn_weights is None:
        print(f"Loading CNN weights from {WEIGHTS_PATH}...")
        cnn_weights = np.load(WEIGHTS_PATH)
        print("CNN weights loaded.")
    if xgb_model is None:
        print(f"Loading XGBoost model from {XGB_PATH}...")
        with open(XGB_PATH, "rb") as f:
            xgb_model = pickle.load(f)
        print("XGBoost model loaded.")

def relu(x):
    return np.maximum(0, x)

def max_pool_2d(x, pool_size=2):
    h, w, c = x.shape
    new_h, new_w = h // pool_size, w // pool_size
    # Reshape to 4D to find max in tiles
    reshaped = x[:new_h * pool_size, :new_w * pool_size, :].reshape(new_h, pool_size, new_w, pool_size, c)
    return reshaped.max(axis=(1, 3))

def conv_2d_valid(x, kernel, bias):
    # x: (h, w, in_c)
    # kernel: (kh, kw, in_c, out_c)
    # bias: (out_c,)
    h, w, in_c = x.shape
    kh, kw, _, out_c = kernel.shape
    new_h, new_w = h - kh + 1, w - kw + 1
    
    out = np.zeros((new_h, new_w, out_c))
    
    # We can use scipy for the 2D convolutions (per channel)
    # Keras Conv2D is equivalent to signal.correlate (no flip)
    # For small filters, we can just use convolve2d on flipped kernels?
    for oc in range(out_c):
        for ic in range(in_c):
            k = kernel[:, :, ic, oc]
            # Keras Conv2D is cross-correlation. scipy.signal.convolve2d with k_flipped = correlation.
            out[:, :, oc] += convolve2d(x[:, :, ic], np.flip(k, axis=(0, 1)), mode='valid')
        out[:, :, oc] += bias[oc]
    
    return out

def extract_features(img):
    """Numpy baseline for the 2-Conv, 2-Pool, 1-Dense CNN."""
    # Input img: (64, 64, 1)
    
    # C1: (3,3,1,16) -> (62, 62, 16)
    t0 = time.time()
    print("CNN: Running Conv2D-1...")
    x = conv_2d_valid(img, cnn_weights['c1_k'], cnn_weights['c1_b'])
    x = relu(x)
    t1 = time.time()
    print(f"CNN: C1 took {t1-t0:.4f}s")
    
    # P1: (62, 62, 16) -> (31, 31, 16)
    x = max_pool_2d(x)
    t2 = time.time()
    print(f"CNN: P1 took {t2-t1:.4f}s")
    
    # C2: (3,3,16,32) -> (29, 29, 32)
    print(f"CNN: Running Conv2D-2 (512 iterations)...")
    x = conv_2d_valid(x, cnn_weights['c2_k'], cnn_weights['c2_b'])
    x = relu(x)
    t3 = time.time()
    print(f"CNN: C2 took {t3-t2:.4f}s")
    
    # P2: (29, 29, 32) -> (14, 14, 32)
    x = max_pool_2d(x)
    t4 = time.time()
    print(f"CNN: P2 took {t4-t3:.4f}s")
    
    # Flatten: (6272,)
    x = x.flatten()
    
    # Dense(32): (6272,) @ (6272, 32) + (32,)
    print("CNN: Running Dense(32)...")
    features = x @ cnn_weights['d1_k'] + cnn_weights['d1_b']
    features = relu(features)
    t5 = time.time()
    print(f"CNN: Dense took {t5-t4:.4f}s")
    
    print(f"CNN: Total extraction time: {t5-t0:.4f}s")
    return features.reshape(1, -1)

def extract_spectrogram(file_path):
    audio, sr = librosa.load(file_path, duration=2.0)
    spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    log_spec = librosa.power_to_db(spec)
    return np.resize(log_spec, (64, 64, 1))

def predict(file_path):
    load_models()
    
    # 1. Prep
    img = extract_spectrogram(file_path)
    
    # 2. Extract features via pure Numpy CNN
    features = extract_features(img)
    
    # 3. XGBoost
    label_idx = int(xgb_model.predict(features)[0])
    probs = xgb_model.predict_proba(features)[0]
    
    result = "FAKE" if label_idx == 1 else "REAL"
    confidence = float(probs[label_idx])
    
    return {
        "result": result,
        "confidence": round(confidence * 100, 2),
        "label": label_idx
    }
