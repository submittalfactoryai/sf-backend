# schemas/auth.py
from pydantic import BaseModel, EmailStr, Field
from schemas.subscription import SubscriptionStatusResponse

class RegisterRequest(BaseModel):
    email: EmailStr
    user_name: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=8)

class RegisterResponse(BaseModel):
    user_id: int
    email: EmailStr
    user_name: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id:int
    email: EmailStr
    user_name: str
    roles: list[str]
    isActive: bool
    canSmartSearch: bool
    canSmartValidate: bool
    subscription: SubscriptionStatusResponse

class AdminPasswordResetRequest(BaseModel):
    user_id: int = Field(..., description="User ID of the account to reset")
    new_password: str = Field(..., min_length=8, max_length=128, description="New password (at least 8 chars)")

class PasswordResetResponse(BaseModel):
    user_id: int
    email: str
    user_name: str
    message: str