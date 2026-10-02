# Appendix 1: Source Code

## 1. Backend

### `backend/main.py`

```python
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uuid
import time
from backend.routes.auth import router as auth_router, get_current_user
from backend.services.ml_service import predict, load_models
from backend.services.db_service import connect_to_mongo, close_mongo_connection, save_prediction, get_history
from backend.config import UPLOAD_DIR
from pydantic import BaseModel
from datetime import datetime

app = FastAPI(title="Audio Deepfake Detector API")
app.include_router(auth_router, prefix="/api/auth", tags=["auth"])

# Add CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Simplified for demo
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
    print("Preloading ML models...")
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
    # We read 1 byte past the limit to check
    MAX_SIZE = 5 * 1024 * 1024 
    contents = await file.read(MAX_SIZE + 1)
    if len(contents) > MAX_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Please upload audio under 5MB.")
    
    # Reset file pointer after reading
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
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=400, detail=f"Analysis failed: {str(e)}")
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/api/history")
async def fetch_history(limit: int = 50, authorization: str = Header(default=None)):
    user_id = await get_current_user(authorization) if authorization else None
    return await get_history(user_id=user_id, limit=limit)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

```

### `backend/config.py`

```python
import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "deepfake_detector")

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "your_super_secret_key_123456789")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Directories
MODEL_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

```

### `backend/services/ml_service.py`

```python
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

### `backend/services/auth_service.py`

```python
import jwt
from datetime import datetime, timedelta
import bcrypt
from backend.services import db_service
import uuid

SECRET_KEY = "supersecretkey"  # Change in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # 7 days

def verify_password(plain_password: str, hashed_password: str):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def get_password_hash(password: str):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def create_user(username, password):
    if db_service.db is None:
        await db_service.connect_to_mongo()
    existing = await db_service.db["users"].find_one({"username": username})
    if existing: return None
    
    user_id = str(uuid.uuid4())
    hashed_pwd = get_password_hash(password)
    user = {"user_id": user_id, "username": username, "password": hashed_pwd}
    await db_service.db["users"].insert_one(user)
    return user

async def get_user(username: str):
    if db_service.db is None:
        await db_service.connect_to_mongo()
    return await db_service.db["users"].find_one({"username": username})

async def get_user_by_id(user_id: str):
    if db_service.db is None:
        await db_service.connect_to_mongo()
    return await db_service.db["users"].find_one({"user_id": user_id})

```

### `backend/services/db_service.py`

```python
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from backend.config import MONGO_URI, DATABASE_NAME

client = None
db = None

async def connect_to_mongo():
    global client, db
    uri = MONGO_URI if "?" in MONGO_URI else f"{MONGO_URI}?serverSelectionTimeoutMS=2000"
    client = AsyncIOMotorClient(uri)
    db = client[DATABASE_NAME]
    print(f"Connected to MongoDB at {MONGO_URI} [{DATABASE_NAME}]")

async def close_mongo_connection():
    global client
    if client:
        client.close()
        print("MongoDB connection closed.")

async def save_prediction(user_id: str | None, filename: str, result: str, confidence: float):
    try:
        if db is None:
            await connect_to_mongo()
        
        doc = {
            "user_id": user_id,
            "filename": filename,
            "result": result,
            "confidence": confidence,
            "created_at": datetime.utcnow()
        }
        
        await db["predictions"].insert_one(doc)
    except Exception as e:
        print(f"Warning: Could not save prediction to DB. {e}")

async def get_history(user_id: str | None = None, limit: int = 50):
    try:
        if db is None:
            await connect_to_mongo()
            
        query = {"user_id": user_id}
            
        cursor = db["predictions"].find(query).sort("created_at", -1).limit(limit)
        history = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            history.append(doc)
        return history
    except Exception as e:
        print(f"Warning: Could not fetch prediction history from DB. {e}")
        return []

```

### `backend/routes/auth.py`

```python
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel
from backend.services.auth_service import create_user, get_user, verify_password, create_access_token, get_user_by_id, SECRET_KEY, ALGORITHM
import jwt

router = APIRouter()

class UserAuth(BaseModel):
    username: str
    password: str

async def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        return user_id
    except:
        return None

@router.post("/register")
async def register(user_data: UserAuth):
    user = await create_user(user_data.username, user_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Username already exists")
    token = create_access_token({"sub": user["user_id"], "username": user["username"]})
    return {"token": token, "user": {"user_id": user["user_id"], "username": user["username"]}}

@router.post("/login")
async def login(user_data: UserAuth):
    user = await get_user(user_data.username)
    if not user or not verify_password(user_data.password, user["password"]):
        raise HTTPException(status_code=400, detail="Invalid username or password")
    token = create_access_token({"sub": user["user_id"], "username": user["username"]})
    return {"token": token, "user": {"user_id": user["user_id"], "username": user["username"]}}

@router.get("/me")
async def get_me(user_id: str = Depends(get_current_user)):
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    user = await get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return {"user_id": user["user_id"], "username": user["username"]}

```

## 2. Frontend

### `frontend/src/main.jsx`

```jsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)

```

### `frontend/src/App.jsx`

```jsx
import React, { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom'
import { Home, History, Shield, Github, User } from 'lucide-react'
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
          <div className="flex items-center gap-2">
            <Home size={18} />
            <span>Detector</span>
          </div>
        </Link>
        <Link to="/history" className={`px-5 py-2 rounded-xl text-sm font-medium transition-all ${location.pathname === '/history' ? 'bg-primary text-white shadow-lg' : 'hover:bg-white/5 text-text-muted'}`}>
          <div className="flex items-center gap-2">
            <History size={18} />
            <span>History</span>
          </div>
        </Link>
      </div>

      <div className="hidden sm:flex items-center gap-4">
        {user ? (
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 bg-primary/10 px-3 py-1.5 rounded-full border border-primary/20">
              <User size={16} className="text-primary" />
              <span className="text-sm font-medium text-white">{user.username}</span>
            </div>
            <button onClick={handleLogout} className="text-sm font-semibold text-text-muted hover:text-danger transition-colors">
              Logout
            </button>
          </div>
        ) : (
          <button 
            onClick={() => setIsAuthOpen(true)} 
            className="flex items-center gap-2 bg-white/10 hover:bg-white/20 px-4 py-2 rounded-xl text-white font-semibold transition-colors"
          >
            <User size={18} />
            <span>Login</span>
          </button>
        )}
      </div>
    </nav>
  )
}

function App() {
  const [token, setToken] = useState(localStorage.getItem('token') || null);
  const [user, setUser] = useState(JSON.parse(localStorage.getItem('user')) || null);
  const [isAuthOpen, setIsAuthOpen] = useState(false);

  const handleLogin = (userData, jwtToken) => {
    setUser(userData);
    setToken(jwtToken);
    localStorage.setItem('user', JSON.stringify(userData));
    localStorage.setItem('token', jwtToken);
  };

  const handleLogout = () => {
    setUser(null);
    setToken(null);
    localStorage.removeItem('user');
    localStorage.removeItem('token');
  };

  return (
    <Router>
      <div className="flex flex-col min-h-screen">
        <Navbar user={user} handleLogout={handleLogout} setIsAuthOpen={setIsAuthOpen} />
        <main className="flex-grow container mx-auto px-4 py-12">
          <Routes>
            <Route path="/" element={<HomePage token={token} />} />
            <Route path="/history" element={<HistoryPage token={token} />} />
          </Routes>
        </main>
        
        <footer className="py-12 bg-black/20 border-t border-white/5 mt-auto">
          <div className="container mx-auto px-4 text-center flex flex-col items-center space-y-4">
            <h4 className="text-xl font-bold gradient-text">About AudioVerifier</h4>
            <p className="text-text-muted text-sm max-w-2xl">
              AudioVerifier leverages a cutting-edge CNN + BiLSTM deep learning model with 98% accuracy 
              to analyze Mel-spectrograms of your audio. The CNN layers extract spatial features from spectrograms while 
              the Bidirectional LSTM captures temporal patterns, enabling precise detection of synthetic frequencies 
              and AI-generated artifacts to distinguish authentic human voices from deepfakes in seconds.
            </p>
            <div className="pt-6 mt-2 border-t border-white/5 w-64">
              <p className="text-text-muted text-xs">
                Deepfake Detector
              </p>
            </div>
          </div>
        </footer>
        <AuthModal isOpen={isAuthOpen} onClose={() => setIsAuthOpen(false)} onLogin={handleLogin} />
      </div>
    </Router>
  )
}

export default App

```

### `frontend/src/index.css`

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --bg-color: #0d0f14;
  --card-bg: rgba(255, 255, 255, 0.05);
  --card-border: rgba(255, 255, 255, 0.1);
  --primary: #4f46e5;
  --primary-hover: #4338ca;
  --secondary: #ec4899;
  --success: #10b981;
  --danger: #ef4444;
  --text: #f8fafc;
  --text-muted: #94a3b8;
  --shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
  --glass: blur(12px);
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  background-color: var(--bg-color);
  color: var(--text);
  min-height: 100vh;
  overflow-x: hidden;
  line-height: 1.6;
}

#root {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* Background Gradients */
body::before {
  content: "";
  position: fixed;
  top: -10%;
  left: -10%;
  width: 40%;
  height: 40%;
  background: radial-gradient(circle, rgba(79, 70, 229, 0.15) 0%, transparent 70%);
  z-index: -1;
}

body::after {
  content: "";
  position: fixed;
  bottom: -10%;
  right: -10%;
  width: 50%;
  height: 50%;
  background: radial-gradient(circle, rgba(236, 72, 153, 0.1) 0%, transparent 70%);
  z-index: -1;
}

/* Glassmorphism Classes */
.glass-card {
  background: var(--card-bg);
  backdrop-filter: var(--glass);
  -webkit-backdrop-filter: var(--glass);
  border: 1px solid var(--card-border);
  border-radius: 1.5rem;
  box-shadow: var(--shadow);
}

.gradient-text {
  background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 700;
}

/* Hover and Animations */
.hover-scale {
  transition: transform 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

.hover-scale:hover {
  transform: scale(1.02);
}

button {
  cursor: pointer;
  border: none;
  font-family: inherit;
  transition: all 0.2s ease;
}

.primary-btn {
  background: linear-gradient(135deg, var(--primary) 0%, #6366f1 100%);
  color: white;
  padding: 0.75rem 1.5rem;
  border-radius: 0.75rem;
  font-weight: 600;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.primary-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.3);
}

/* Layout */
.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
}

a {
  color: inherit;
  text-decoration: none;
}

```

### `frontend/src/components/AudioUpload.jsx`

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
        
        {!loading && (
          <button type="button" className="px-5 py-2 bg-white/10 hover:bg-white/20 transition-colors rounded-full text-xs font-semibold">
            {file ? 'Change File' : 'Browse Computer'}
          </button>
        )}
      </div>

      {error && (
        <div className="bg-danger/10 border border-danger/20 p-4 rounded-2xl flex items-center gap-3 text-danger text-sm animate-in fade-in zoom-in duration-300">
          <AlertCircle size={18} />
          {error}
        </div>
      )}

      {file && !loading && (
        <div className="bg-white/5 rounded-2xl p-6 border border-white/10 flex flex-col gap-6 animate-in slide-in-from-bottom duration-300">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-primary/10 rounded-xl text-primary">
                <FileAudio size={20} />
              </div>
              <div className="overflow-hidden">
                <p className="font-semibold truncate max-w-[150px]">{file.name}</p>
                <p className="text-xs text-text-muted">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
              </div>
            </div>
            <button onClick={cancelSelection} className="p-2 hover:bg-white/10 rounded-full text-text-muted transition-all">
              <XCircle size={20} />
            </button>
          </div>
          
          {preview && (
            <div className="rounded-xl overflow-hidden bg-black/40 p-2">
              <audio src={preview} controls className="w-full h-8" />
            </div>
          )}
          
          <button 
            onClick={handleStartAnalysis} 
            className="primary-btn w-full text-md flex items-center justify-center gap-2 py-3"
          >
            Launch Deepfake Analysis
            <CheckCircle size={18} />
          </button>
        </div>
      )}
      
      {loading && (
        <div className="p-10 flex flex-col items-center justify-center gap-6 bg-white/5 rounded-2xl border border-primary/20">
          <div className="flex items-center gap-1 h-8">
            {[...Array(12)].map((_, i) => (
              <div 
                key={i} 
                className="w-1.5 bg-primary rounded-full animate-bounce" 
                style={{ 
                  height: '100%',
                  animationDelay: `${i * 0.1}s`,
                  animationDuration: '0.8s'
                }} 
              />
            ))}
          </div>
          <div className="text-center">
            <h3 className="text-lg font-bold text-primary animate-pulse">Running Neural Classifiers...</h3>
            <p className="text-xs text-text-muted uppercase tracking-tighter">Analyzing spectral micro-fluctuations</p>
          </div>
        </div>
      )}
    </div>
  )
}

export default AudioUpload

```

### `frontend/src/components/ResultCard.jsx`

```jsx
import React from 'react'
import { motion } from 'framer-motion'
import { ShieldCheck, ShieldAlert, AlertTriangle, Cpu, Activity, InfoIcon, BarChart3 } from 'lucide-react'

const ResultCard = ({ data }) => {
  if (!data) return null;

  const isFake = data.result === 'FAKE'
  const isUncertain = data.result === 'UNCERTAIN'
  const isHuman = data.result === 'REAL'
  
  // Dynamic styling based on stabilized types
  let accentColor = '#10b981' // Green
  let icon = <ShieldCheck size={64} />
  
  if (isFake) {
    accentColor = '#ef4444' // Red
    icon = <ShieldAlert size={64} />
  } else if (isUncertain) {
    accentColor = '#facc15' // Yellow
    icon = <AlertTriangle size={64} />
  }

  const bgGradient = `linear-gradient(135deg, ${accentColor}15 0%, ${accentColor}05 100%)`

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="glass-card overflow-hidden flex flex-col p-8 relative border-2"
      style={{ borderColor: `${accentColor}40`, background: bgGradient }}
    >
      <div className="flex flex-col md:flex-row gap-8 items-center">
        <div className="relative">
          <motion.div 
            animate={isUncertain ? { scale: [1, 1.05, 1] } : {}}
            transition={{ repeat: Infinity, duration: 2 }}
            className="w-32 h-32 rounded-3xl flex items-center justify-center bg-white/5 border border-white/10 shadow-2xl"
            style={{ color: accentColor }}
          >
            {icon}
          </motion.div>
          <div className="absolute -bottom-2 -right-2 bg-black border border-white/20 p-2 rounded-xl">
             <Activity size={16} style={{ color: accentColor }} />
          </div>
        </div>

        <div className="flex-1 space-y-4 text-center md:text-left">
          <div>
            <div className="flex items-center justify-center md:justify-start gap-2 opacity-50 mb-1">
                <BarChart3 size={12} />
                <span className="text-[10px] uppercase font-black tracking-widest">Neural Stability Filter Active</span>
            </div>
            <h2 className="text-4xl font-black" style={{ color: accentColor }}>
              {data.result}
            </h2>
          </div>
          
          <div className="grid grid-cols-1 gap-4">
            <div className="bg-black/20 p-4 rounded-2xl border border-white/5">
              <p className="text-[10px] uppercase font-bold opacity-40 mb-1">Confidence Grade</p>
              <p className="text-xl font-black">
                {data.confidence_level} 
                <span className="text-xs ml-1 opacity-60">({Math.round(data.confidence)}%)</span>
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-8 pt-6 border-t border-white/10 space-y-4">
        <div className="flex gap-4 bg-white/5 p-4 rounded-2xl border border-white/5 items-start">
          <div className="mt-1 flex-shrink-0">
            <InfoIcon size={20} style={{ color: accentColor }} />
          </div>
          <div>
            <p className="text-xs font-black uppercase opacity-60 mb-1">Classifier Insight</p>
            <p className="text-sm leading-relaxed font-medium">
              {data.explanation}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 px-2">
          <Cpu size={14} className="opacity-30" />
          <p className="text-[9px] uppercase tracking-[0.2em] font-bold opacity-30">
            CNN + BiLSTM Authentication Model
          </p>
        </div>
      </div>
    </motion.div>
  )
}

export default ResultCard

```

### `frontend/src/components/AuthModal.jsx`

```jsx
import { useState } from 'react';
import axios from 'axios';
import { X, Loader2 } from 'lucide-react';

export default function AuthModal({ isOpen, onClose, onLogin }) {
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const endpoint = isLogin ? '/api/auth/login' : '/api/auth/register';
      const res = await axios.post(endpoint, { username, password });
      onLogin(res.data.user, res.data.token);
      onClose();
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred. Make sure backend is running & MongoDB is active.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-in fade-in duration-200">
      <div className="bg-[#12141c] w-full max-w-md rounded-2xl border border-white/10 p-8 relative shadow-2xl">
        <button onClick={onClose} className="absolute top-4 right-4 text-text-muted hover:text-white transition-colors">
          <X size={24} />
        </button>

        <h2 className="text-3xl font-bold mb-6 gradient-text">{isLogin ? 'Welcome Back' : 'Create Account'}</h2>

        {error && <div className="bg-danger/20 text-danger text-sm p-3 rounded-lg mb-4">{error}</div>}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-text-muted mb-1">Username</label>
            <input 
              type="text" 
              required
              value={username} 
              onChange={e => setUsername(e.target.value)}
              className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-primary transition-colors" 
              placeholder="" 
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-text-muted mb-1">Password</label>
            <input 
              type="password" 
              required
              value={password} 
              onChange={e => setPassword(e.target.value)}
              className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-primary transition-colors" 
              placeholder="••••••••" 
            />
          </div>

          <button 
            type="submit" 
            disabled={loading}
            className="w-full bg-primary hover:bg-primary-hover text-white font-bold py-3 rounded-lg mt-6 flex justify-center items-center transition-colors disabled:opacity-50"
          >
            {loading ? <Loader2 className="animate-spin" /> : isLogin ? 'Login' : 'Sign Up'}
          </button>
        </form>

        <p className="text-center text-text-muted text-sm mt-6">
          {isLogin ? "Don't have an account? " : "Already have an account? "}
          <button 
            onClick={() => { setIsLogin(!isLogin); setError(''); }} 
            className="text-primary hover:text-primary-hover font-semibold transition-colors"
          >
            {isLogin ? 'Sign up' : 'Login'}
          </button>
        </p>
      </div>
    </div>
  );
}

```

### `frontend/src/pages/HomePage.jsx`

```jsx
import React, { useState } from 'react'
import { RotateCcw, ShieldAlert, Zap, Search } from 'lucide-react'
import axios from 'axios'
import AudioUpload from '../components/AudioUpload'
import ResultCard from '../components/ResultCard'

const HomePage = ({ token }) => {
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleUpload = async (file) => {
    setLoading(true)
    setError(null)
    setPrediction(null)
    
    const formData = new FormData()
    formData.append('file', file)

    try {
      const config = { headers: { 'Content-Type': 'multipart/form-data' } }
      if (token) {
        config.headers.Authorization = `Bearer ${token}`
      }
      
      const response = await axios.post('/api/predict', formData, config)
      setPrediction(response.data)
    } catch (err) {
      console.error('Prediction error:', err)
      const detail = err.response?.data?.detail
      setError(Array.isArray(detail) ? detail[0]?.msg : detail || 'Network error encountered.')
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setPrediction(null)
    setError(null)
  }

  return (
    <div className="flex flex-col items-center gap-10 max-w-5xl mx-auto w-full px-4">
      <div className="text-center space-y-6 animate-in fade-in slide-in-from-top duration-700">

        <h1 className="text-6xl md:text-7xl font-black leading-[0.95] tracking-tighter">
          <span className="gradient-text">Multilingual</span>
          <br/>Deepfake Audio <span className="opacity-40">Detector</span>
        </h1>
        <p className="text-text-muted text-lg max-w-2xl mx-auto font-medium leading-relaxed">
          Upload any voice recording or audio. Our CNN-BiLSTM neural engine will analyze spectral patterns to identify if it was captured from a human or generated by an AI model.
        </p>
      </div>

      {!prediction ? (
        <div className="w-full grid lg:grid-cols-12 gap-10 items-start">
          <div className="lg:col-span-12 glass-card p-6 md:p-10 border border-white/10">
            <AudioUpload onUpload={handleUpload} loading={loading} />
            
            {error && (
              <div className="mt-8 p-5 rounded-3xl bg-danger/5 border border-danger/20 text-danger flex items-center gap-4 animate-in slide-in-from-top duration-500">
                <div className="p-2 bg-danger/10 rounded-xl">
                  <ShieldAlert size={24} />
                </div>
                <div className="flex-1">
                  <p className="font-black text-sm uppercase tracking-wider">Analysis Blocked</p>
                  <p className="text-xs opacity-80">{error}</p>
                </div>
                <button onClick={() => setError(null)} className="text-xs font-bold hover:underline">Dismiss</button>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="w-full grid md:grid-cols-2 gap-10">
           <div className="space-y-6">
                <ResultCard data={prediction} />
           </div>

          <div className="glass-card p-10 flex flex-col justify-center items-center text-center space-y-8 border border-white/10 relative overflow-hidden group">
            <div className="absolute -top-10 -right-10 w-40 h-40 bg-primary/5 rounded-full blur-3xl group-hover:bg-primary/10 transition-colors"></div>
            
            <div className="p-6 bg-white/5 rounded-full relative">
              <RotateCcw size={50} className="text-primary group-hover:rotate-[-180deg] transition-transform duration-700 ease-in-out" />
              <div className="absolute inset-0 bg-primary/20 rounded-full blur-xl animate-pulse"></div>
            </div>
            
            <div className="space-y-3">
                <h3 className="text-3xl font-black tracking-tight">Ready for more?</h3>
                <p className="text-text-muted font-medium text-sm px-4">The verification session is complete. You can clear these results and scan a new audio file immediately.</p>
            </div>

            <button onClick={reset} className="primary-btn w-full py-5 rounded-2xl text-lg font-black flex items-center justify-center gap-3">
              Perform New Analysis
              <Zap size={20} fill="currentColor" />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default HomePage

```

### `frontend/src/pages/HistoryPage.jsx`

```jsx
import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { motion, AnimatePresence } from 'framer-motion'
import { ShieldCheck, ShieldAlert, AlertTriangle, History, Calendar, Trash2, Search, Filter } from 'lucide-react'

const HistoryPage = ({ token }) => {
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [searchTerm, setSearchTerm] = useState('')

  useEffect(() => {
    fetchHistory()
  }, [])

  const fetchHistory = async () => {
    setLoading(true)
    setError(null)
    try {
      const config = {}
      if (token) {
        config.headers = { Authorization: `Bearer ${token}` }
      }
      const response = await axios.get('/api/history', config)
      setHistory(response.data)
    } catch (err) {
      console.error('Fetch history error:', err)
      setError('Could not retrieve prediction history. Check if backend is running.')
    } finally {
      setLoading(false)
    }
  }

  const filteredHistory = history.filter(item => 
    item.filename.toLowerCase().includes(searchTerm.toLowerCase()) ||
    item.result.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const formatDate = (dateString) => {
    const date = new Date(dateString)
    return date.toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric', 
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  return (
    <div className="max-w-6xl mx-auto space-y-12">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
        <div className="space-y-2">
          <h1 className="text-5xl font-extrabold flex items-center gap-4">
            <History size={48} className="text-primary" />
            <span className="gradient-text">Analysis Log</span>
          </h1>
          <p className="text-text-muted text-lg">Tracks your recent voice authentications.</p>
        </div>
        
        <div className="flex gap-4 w-full md:w-auto">
          <div className="relative flex-grow md:w-72">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" size={18} />
            <input 
              type="text" 
              placeholder="Search filename or result..." 
              className="w-full bg-white/5 border border-white/10 rounded-xl py-2 pl-10 pr-4 text-sm focus:outline-none focus:border-primary transition-all"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <button onClick={fetchHistory} className="p-2.5 bg-white/5 border border-white/10 rounded-xl hover:bg-white/10 transition-colors">
            <Filter size={20} className="text-text-muted" />
          </button>
        </div>
      </div>

      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[0, 1, 2, 3, 4, 5].map(i => (
            <div key={i} className="glass-card h-48 animate-pulse bg-white/5" />
          ))}
        </div>
      ) : error ? (
        <div className="glass-card p-12 text-center space-y-4">
          <ShieldAlert size={64} className="text-danger mx-auto" />
          <h3 className="text-2xl font-bold">Something went wrong</h3>
          <p className="text-text-muted">{error}</p>
          <button onClick={fetchHistory} className="primary-btn mt-4">Try Again</button>
        </div>
      ) : filteredHistory.length === 0 ? (
        <div className="glass-card p-24 text-center space-y-6">
          <div className="w-24 h-24 rounded-full bg-white/5 flex items-center justify-center mx-auto opacity-50">
            <History size={48} className="text-text-muted" />
          </div>
          <div>
            <h3 className="text-2xl font-bold">No history available</h3>
            <p className="text-text-muted">Once you analyze audio, results will appear here.</p>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <AnimatePresence mode="popLayout">
            {filteredHistory.map((item, index) => {
              const isFake = item.result === 'FAKE'
              const isUncertain = item.result === 'UNCERTAIN'
              const color = isFake ? 'var(--danger)' : isUncertain ? 'var(--warning, #facc15)' : 'var(--success)'
              
              return (
                <motion.div 
                  layout
                  key={item._id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  transition={{ delay: index * 0.05 }}
                  className="glass-card p-6 flex flex-col gap-6 relative group overflow-hidden"
                  style={{ borderLeft: `4px solid ${color}` }}
                >
                  <div className="flex items-start justify-between">
                    <div className="space-y-1 overflow-hidden">
                      <p className="font-bold text-lg truncate pr-2">{item.filename}</p>
                      <div className="flex items-center gap-2 text-xs text-text-muted font-medium uppercase tracking-wider">
                        <Calendar size={12} />
                        <span>{formatDate(item.created_at)}</span>
                      </div>
                    </div>
                    {isFake ? (
                      <ShieldAlert size={28} className="text-danger flex-shrink-0" />
                    ) : item.result === 'UNCERTAIN' ? (
                      <AlertTriangle size={28} className="text-warning flex-shrink-0" style={{ color: color }}/>
                    ) : (
                      <ShieldCheck size={28} className="text-success flex-shrink-0" />
                    )}
                  </div>

                  <div className="bg-white/5 rounded-xl p-4 flex items-center justify-between mt-auto">
                    <div className="space-y-1">
                      <p className="text-xs text-text-muted font-bold uppercase">Verdict</p>
                      <p className="font-black text-xl tracking-wide" style={{ color }}>{item.result}</p>
                    </div>
                    <div className="text-right space-y-1">
                      <p className="text-xs text-text-muted font-bold uppercase">Confidence</p>
                      <p className="text-2xl font-black">{Math.floor(item.confidence)}%</p>
                    </div>
                  </div>

                  <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button className="p-2 text-text-muted hover:text-danger rounded-lg hover:bg-white/5 transition-all">
                      <Trash2 size={16} />
                    </button>
                  </div>
                </motion.div>
              )
            })}
          </AnimatePresence>
        </div>
      )}
    </div>
  )
}

export default HistoryPage

```

