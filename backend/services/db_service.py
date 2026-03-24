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
