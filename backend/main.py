import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uuid
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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictResponse(BaseModel):
    id: str
    filename: str
    result: str
    confidence: float
    timestamp: datetime

class HistoryItem(BaseModel):
    _id: str
    filename: str
    result: str
    confidence: float
    created_at: datetime

@app.on_event("startup")
async def startup_event():
    # Preload ML models in memory
    print("Preloading ML models...")
    load_models()
    # Connect to MongoDB
    await connect_to_mongo()

@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()

@app.get("/")
async def health_check():
    return {"status": "ok", "service": "Audio Deepfake Detection Backend"}

@app.post("/api/predict", response_model=PredictResponse)
async def predict_audio(file: UploadFile = File(...), authorization: str = Header(default=None)):
    # 1. Validation
    if not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
        raise HTTPException(status_code=400, detail="Invalid audio format")
    
    # 2. Save temporary file
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1]
    temp_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 3. ML Inference
        result_data = predict(temp_path)
        
        user_id = await get_current_user(authorization) if authorization else None
        
        # 4. Save to Database
        await save_prediction(
            user_id=user_id, # Link dynamically!
            filename=file.filename,
            result=result_data["result"],
            confidence=result_data["confidence"]
        )
        
        return {
            "id": file_id,
            "filename": file.filename,
            "result": result_data["result"],
            "confidence": result_data["confidence"],
            "timestamp": datetime.now()
        }
    
    finally:
        # 5. Clean up the temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/api/history")
async def fetch_history(limit: int = 50, authorization: str = Header(default=None)):
    user_id = await get_current_user(authorization) if authorization else None
    return await get_history(user_id=user_id, limit=limit)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
