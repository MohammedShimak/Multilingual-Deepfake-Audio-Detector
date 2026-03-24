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
