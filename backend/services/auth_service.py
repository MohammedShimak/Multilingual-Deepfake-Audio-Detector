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
