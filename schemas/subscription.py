from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Optional,List

class UserSubscriptionData(BaseModel):
    user_id: int
    subscription_type: str
    is_active: bool
    api_calls_used: int
    api_call_limit: int
    api_calls_remaining: int
    is_expired: bool
    is_locked: bool
    valid_until: Optional[str]
    expiry_date: Optional[str]
    days_remaining: Optional[int]
    admin_contact_email: str

class UserWithSubscription(BaseModel):
    id: int
    name: str
    email: str
    is_active: bool
    subscription: Optional[UserSubscriptionData]

class UsersWithSubscriptionsResponse(BaseModel):
    users: List[UserWithSubscription]
    total: int
    page: int
    limit: int
    has_more: bool

class SubscriptionBase(BaseModel):
    subscription_type: str = Field(
        ..., 
        description="Type: free_trial, limited, unlimited, expired"
    )
    api_call_limit: int = Field(
        ..., 
        ge=-1, 
        description="-1 for unlimited PDF uploads, >=0 for limited"
    )
    valid_until: Optional[datetime] = None


class SubscriptionCreate(SubscriptionBase):
    user_id: int


class SubscriptionResponse(SubscriptionBase):
    subscription_id: int
    user_id: int
    api_calls_used: int
    api_calls_remaining: int
    valid_from: datetime
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class SubscriptionGrantRequest(BaseModel):
    user_id: int
    subscription_type: str = Field(
        ..., 
        description="limited or unlimited"
    )
    api_call_limit: Optional[int] = Field(
        None, 
        ge=-1,
        description="Number of PDF uploads allowed. -1 for unlimited"
    )
    valid_days: Optional[int] = Field(
        None, 
        ge=1, 
        description="Number of days subscription is valid"
    )
    
    @validator('subscription_type')
    def validate_type(cls, v):
        allowed = ['limited', 'unlimited']
        if v not in allowed:
            raise ValueError(f'subscription_type must be one of {allowed}')
        return v


class SubscriptionStatusResponse(BaseModel):
    is_active: bool
    subscription_type: str
    api_calls_used: int
    api_call_limit: int
    api_calls_remaining: int
    is_expired: bool
    is_locked: bool
    admin_contact_email: str
    valid_until: Optional[str] = None
    display_message: str  # **NEW: User-friendly message**


class SystemSettingResponse(BaseModel):
    setting_key: str
    setting_value: str
    description: Optional[str] = None
    
    class Config:
        from_attributes = True


class SystemSettingUpdate(BaseModel):
    setting_value: str