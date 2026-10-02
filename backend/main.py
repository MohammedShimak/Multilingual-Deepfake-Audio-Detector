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
