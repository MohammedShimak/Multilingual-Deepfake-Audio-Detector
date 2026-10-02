# AudioVerifier: Concise Production Source Code

This document gathers the essential, modular source code files of the **AudioVerifier Audio Deepfake Detector** project in a clean, easily readable format.

---

## ⚙️ Section 1: FastAPI Asynchronous Backend

### 1.1 `backend/main.py`
The API orchestrator setting up startup handlers, CORS settings, database logs, and the relaxed prediction gateway.

```python
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uuid
from backend.routes.auth import router as auth_router, get_current_user
from backend.services.ml_service import predict, load_models
from backend.services.db_service import connect_to_mongo, close_mongo_connection, save_prediction, get_history
from backend.config import UPLOAD_DIR
from pydantic import BaseModel
from datetime import datetime

app = FastAPI(title="Audio Deepfake Detector API")
app.include_router(auth_router, prefix="/api/auth", tags=["auth"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictResponse(BaseModel):
    id: str
    filename: str
    result: str
    confidence: float
    probability: float
    explanation: str
    timestamp: datetime
    status: str
    confidence_level: str

@app.on_event("startup")
async def startup_event():
    load_models()
    await connect_to_mongo()

@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()

@app.get("/")
async def health_check():
    return {"status": "ok", "service": "Audio Deepfake Detection Backend"}

@app.post("/api/predict", response_model=PredictResponse)
async def predict_audio(file: UploadFile = File(...), authorization: str = Header(default=None)):
    # 1. Validation: Multiple Formats Supported
    ALLOWED_EXTENSIONS = ('.wav', '.mp3', '.ogg', '.flac', '.aac', '.m4a', '.webm')
    has_allowed_extension = file.filename.lower().endswith(ALLOWED_EXTENSIONS)
    is_audio_mime = file.content_type and file.content_type.startswith("audio/")
    has_no_extension = "." not in file.filename
    
    if not (has_allowed_extension or is_audio_mime or has_no_extension):
        raise HTTPException(status_code=400, detail=f"Unsupported format. Please upload {', '.join(ALLOWED_EXTENSIONS)} files.")
    
    # 2. Validation: File Size < 5MB
    MAX_SIZE = 5 * 1024 * 1024 
    contents = await file.read(MAX_SIZE + 1)
    if len(contents) > MAX_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Please upload audio under 5MB.")
    
    await file.seek(0)
    
    # 3. Save temporary file
    file_id = str(uuid.uuid4())
    temp_path = os.path.join(UPLOAD_DIR, f"{file_id}.wav")
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 4. ML Inference
        result_data = predict(temp_path)
        user_id = await get_current_user(authorization) if authorization else None
        
        # 5. Save to Database
        await save_prediction(
            user_id=user_id,
            filename=file.filename,
            result=result_data["result"],
            confidence=result_data["confidence"]
        )
        
        return {
            "id": file_id,
            "filename": file.filename,
            "result": result_data["result"],
            "confidence": result_data["confidence"],
            "probability": result_data["probability"],
            "explanation": result_data["explanation"],
            "status": result_data["status"],
            "confidence_level": result_data["confidence_level"],
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/api/history")
async def fetch_history(limit: int = 50, authorization: str = Header(default=None)):
    user_id = await get_current_user(authorization) if authorization else None
    return await get_history(user_id=user_id, limit=limit)
```

### 1.2 `backend/services/ml_service.py`
The signals processing service performing MFCC feature engineering and running static Keras prediction layers.

```python
import os
import time
import librosa
import numpy as np
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model = None
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_PATH, "deepfake_audio_20k_model.keras")

def load_models():
    global model
    if model is not None:
        return
    try:
        model = load_model(MODEL_PATH, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")

def extract_features(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    target_length = 823
    
    if mfccs.shape[1] < target_length:
        pad_width = target_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0,0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :target_length]
        
    mfccs = mfccs.T  # Transpose to (823, 40)
    mfccs = np.expand_dims(mfccs, axis=0)  # (1, 823, 40)
    return mfccs

def predict(file_path):
    load_models()
    if model is None:
        raise RuntimeError("Model not available.")

    duration = librosa.get_duration(path=file_path)
    if duration < 1.0:
        raise ValueError("Audio too short for analysis (min 1.0s).")

    audio, sr = librosa.load(file_path, sr=16000)
    
    # Amplitude normalization
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / (max_val + 1e-6)
    else:
        raise ValueError("Silent audio file.")
    
    features = extract_features(audio, sr)
    prob = float(model.predict(features, verbose=0)[0][0])

    if prob > 0.5:
        result = "FAKE"
        confidence = prob * 100.0
        confidence_level = "High" if prob > 0.8 else "Medium"
        explanation = "Synthesized markers detected. Classified as FAKE."
    else:
        result = "REAL"
        confidence = (1.0 - prob) * 100.0
        confidence_level = "High" if prob < 0.2 else "Medium"
        explanation = "Authentic voice patterns confirmed. Classified as REAL."

    return {
        "result": result,
        "confidence": round(confidence, 2),
        "confidence_level": confidence_level,
        "probability": prob,
        "explanation": explanation,
        "status": "success"
    }
```

---

## 💻 Section 2: React Frontend App

### 2.1 `frontend/src/components/AudioUpload.jsx`
Drag and drop component providing instant selection and real-time client validation checks.

```jsx
import React, { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, FileAudio, CheckCircle, XCircle, Loader2, AlertCircle } from 'lucide-react'

const AudioUpload = ({ onUpload, loading }) => {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [error, setError] = useState(null)

  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    setError(null)
    
    if (rejectedFiles?.length > 0) {
      const reason = rejectedFiles[0].errors[0].code
      if (reason === 'file-invalid-type') {
        setError("Unsupported format. Please upload WAV, MP3, OGG, or FLAC.")
      } else if (reason === 'file-too-large') {
        setError("File is too large. Please upload audio under 5MB.")
      } else {
        setError("Invalid file selection.")
      }
      return
    }

    if (acceptedFiles?.length > 0) {
      const selectedFile = acceptedFiles[0]
      setFile(selectedFile)
      if (preview) URL.revokeObjectURL(preview)
      setPreview(URL.createObjectURL(selectedFile))
    }
  }, [preview])

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: { 
      'audio/wav': ['.wav'],
      'audio/mpeg': ['.mp3'],
      'audio/ogg': ['.ogg'],
      'audio/flac': ['.flac'],
      'audio/x-m4a': ['.m4a']
    },
    multiple: false,
    maxSize: 5242880, // 5MB
    disabled: loading
  })

  const handleStartAnalysis = () => {
    if (file && onUpload) {
      onUpload(file)
    }
  }

  const cancelSelection = () => {
    setFile(null)
    if (preview) URL.revokeObjectURL(preview)
    setPreview(null)
    setError(null)
  }

  return (
    <div className="flex flex-col gap-8 w-full">
      <div 
        {...getRootProps()} 
        className={`relative border-2 border-dashed rounded-3xl p-10 transition-all flex flex-col items-center gap-6 cursor-pointer group 
        ${isDragActive ? 'border-primary bg-primary/5 scale-[1.01]' : 'border-white/10 hover:border-white/20 hover:bg-white/5'}
        ${isDragReject ? 'border-danger bg-danger/5' : ''}
        ${loading ? 'opacity-50 pointer-events-none' : ''}`}
      >
        <input {...getInputProps()} />
        <div className="w-16 h-16 bg-primary/10 rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform">
          {loading ? (
            <Loader2 className="animate-spin text-primary" size={32} />
          ) : file ? (
            <CheckCircle className="text-success" size={32} />
          ) : (
            <Upload className="text-primary" size={32} />
          )}
        </div>
        <div className="text-center space-y-1">
          <p className="text-xl font-bold">
            {isDragActive ? 'Drop here' : file ? 'Audio Selected' : 'Select voice sample'}
          </p>
          <p className="text-sm text-text-muted">
            {file ? file.name : 'Supports WAV, MP3, OGG, & FLAC (<5MB)'}
          </p>
        </div>
      </div>

      {error && (
        <div className="bg-danger/10 border border-danger/20 p-4 rounded-2xl flex items-center gap-3 text-danger text-sm">
          <AlertCircle size={18} />
          {error}
        </div>
      )}

      {file && !loading && (
        <div className="bg-white/5 rounded-2xl p-6 border border-white/10 flex flex-col gap-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-primary/10 rounded-xl text-primary">
                <FileAudio size={20} />
              </div>
              <div className="overflow-hidden">
                <p className="font-semibold truncate max-w-[250px]">{file.name}</p>
                <p className="text-xs text-text-muted">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
              </div>
            </div>
            <button onClick={cancelSelection} className="p-2 hover:bg-white/10 rounded-full text-text-muted">
              <XCircle size={20} />
            </button>
          </div>
          {preview && (
            <div className="rounded-xl overflow-hidden bg-black/40 p-2">
              <audio src={preview} controls className="w-full h-8" />
            </div>
          )}
          <button onClick={handleStartAnalysis} className="primary-btn w-full py-3">
            Launch Deepfake Analysis
          </button>
        </div>
      )}
    </div>
  )
}

export default AudioUpload;
```

### 2.2 `frontend/src/App.jsx`
Main client setup routing navigation, state holders, and footer description components.

```jsx
import React, { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom'
import { Home, History, Shield, User } from 'lucide-react'
import HomePage from './pages/HomePage'
import HistoryPage from './pages/HistoryPage'
import AuthModal from './components/AuthModal'

const Navbar = ({ user, handleLogout, setIsAuthOpen }) => {
  const location = useLocation()
  return (
    <nav className="glass-card mt-6 mx-4 px-6 py-4 flex items-center justify-between sticky top-6 z-50">
      <div className="flex items-center gap-3">
        <div className="bg-primary p-2 rounded-xl">
          <Shield color="white" size={24} />
        </div>
        <span className="text-xl font-bold gradient-text">AudioVerifier</span>
      </div>
      <div className="flex bg-white/5 rounded-2xl p-1 gap-2 border border-white/10">
        <Link to="/" className={`px-5 py-2 rounded-xl text-sm font-medium transition-all ${location.pathname === '/' ? 'bg-primary text-white shadow-lg' : 'hover:bg-white/5 text-text-muted'}`}>
          Detector
        </Link>
        <Link to="/history" className={`px-5 py-2 rounded-xl text-sm font-medium transition-all ${location.pathname === '/history' ? 'bg-primary text-white shadow-lg' : 'hover:bg-white/5 text-text-muted'}`}>
          History
        </Link>
      </div>
    </nav>
  )
}

function App() {
  const [token, setToken] = useState(localStorage.getItem('token') || null);
  const [user, setUser] = useState(JSON.parse(localStorage.getItem('user')) || null);
  const [isAuthOpen, setIsAuthOpen] = useState(false);

  return (
    <Router>
      <div className="flex flex-col min-h-screen">
        <Navbar user={user} setIsAuthOpen={setIsAuthOpen} />
        <main className="flex-grow container mx-auto px-4 py-12">
          <Routes>
            <Route path="/" element={<HomePage token={token} />} />
            <Route path="/history" element={<HistoryPage token={token} />} />
          </Routes>
        </main>
        <footer className="py-12 bg-black/20 border-t border-white/5 mt-auto">
          <div className="container mx-auto px-4 text-center">
            <h4 className="text-xl font-bold gradient-text">About AudioVerifier</h4>
            <p className="text-text-muted text-sm max-w-2xl mx-auto mt-2">
              AudioVerifier leverages a cutting-edge CNN + Gradient Boosting hybrid machine learning model 
              to analyze Mel-spectrograms of your audio, enabling precise detection of AI-generated signatures.
            </p>
          </div>
        </footer>
      </div>
    </Router>
  )
}

export default App;
```
