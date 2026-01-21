# src/schemas/user.py
from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional, List, Dict, Any

class UserResponse(BaseModel):
    id: int
    name: str
    email: EmailStr
    role: str
    isActive: bool
    lastLogin: Optional[datetime]
    canSmartSearch: bool
    canSmartValidate: bool
    roles: List[str] 

class UserActivationRequest(BaseModel):
    is_active: bool

from pydantic import BaseModel, EmailStr

class UserUpdateRequest(BaseModel):
    name: Optional[str]
    email: Optional[EmailStr]
    role: str  # "admin" or "user"
    canSmartSearch: Optional[bool] = False
    canSmartValidate: Optional[bool] = False

class CustomAuditLogRequest(BaseModel):
    action: str                  # e.g., "PreviewPDF", "DownloadStarted", etc.
    entity_type: str             # e.g., "PDF", "User", "Batch"
    entity_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None  # Arbitrary data about the action
    cost: Optional[float] = None               