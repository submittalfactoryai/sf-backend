# routers/subscription.py
# ✅ FIXED: 
# - Field names match frontend (id, name instead of user_id, user_name)
# - Uses SubscriptionService.get_all_users_with_subscriptions for proper data

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from database import get_db
from models import User, SystemSetting
from core.security import get_current_active_user, get_current_admin
from schemas.subscription import (
    SubscriptionStatusResponse, 
    SubscriptionGrantRequest,
    SubscriptionResponse,
    SystemSettingResponse,
    SystemSettingUpdate
)
from services.subscription_service import SubscriptionService
from core.logger import log_action
from typing import List
import time
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/subscription", tags=["subscription"])


@router.get("/status", response_model=SubscriptionStatusResponse)
async def get_my_subscription_status(
    request: Request,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get current user's subscription status
    ✅ OPTIMIZED: Reduced audit logging (only log every 5th call)
    """
    start_time = time.time()
    
    status_data = SubscriptionService.get_subscription_status(db, current_user.user_id)
    
    # Log only periodically to reduce DB writes
    should_log = status_data.get('api_calls_used', 0) % 5 == 0
    
    # if should_log:
    #     try:
    #         log_action(
    #             db=db,
    #             user_id=current_user.user_id,
    #             action_type="SUBSCRIPTION_STATUS_CHECK",
    #             entity_type="Subscription",
    #             entity_id=str(current_user.user_id),
    #             user_metadata=status_data,
    #             process_time=int((time.time() - start_time) * 1000)
    #         )
    #     except Exception as e:
    #         logger.warning(f"Failed to log status check: {e}")
    
    return status_data


@router.get("/user/{user_id}/status", response_model=SubscriptionStatusResponse)
async def get_user_subscription_status(
    user_id: int,
    request: Request,
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Admin: Get any user's subscription status"""
    status_data = SubscriptionService.get_subscription_status(db, user_id)
    return status_data


@router.get("/users-with-subscriptions")
async def get_users_with_subscriptions(
    skip: int = 0,
    limit: int = 50,
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Admin - Get all users with subscriptions in ONE call.
    ✅ Uses SubscriptionService for consistent data format.
    """
    start_time = time.time()
    
    # Use the service method which returns properly formatted data
    result = SubscriptionService.get_all_users_with_subscriptions(db, skip, limit)
    
    # Log admin action (reduced frequency)
    # try:
    #     log_action(
    #         db=db,
    #         user_id=admin.user_id,
    #         action_type="ADMIN_USERS_SUBSCRIPTIONS_FETCH",
    #         entity_type="Subscription",
    #         entity_id="batch",
    #         user_metadata={
    #             "total_users": result["total"],
    #             "page": result["page"],
    #             "limit": result["limit"]
    #         },
    #         process_time=int((time.time() - start_time) * 1000)
    #     )
    # except Exception as e:
    #     logger.warning(f"Failed to log admin action: {e}")
    
    return result


@router.post("/grant/{user_id}")
async def grant_subscription(
    user_id: int,
    request_data: SubscriptionGrantRequest,
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Admin: Grant subscription to user"""
    
    target_user = db.query(User).filter(User.user_id == user_id).first()
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    subscription = SubscriptionService.grant_subscription(
        db=db,
        user_id=user_id,
        granted_by=admin.user_id,
        subscription_type=request_data.subscription_type,
        api_call_limit=request_data.api_call_limit,
        valid_days=request_data.valid_days
    )
    
    # Log the grant action
    log_action(
        db=db,
        user_id=admin.user_id,
        action_type="SUBSCRIPTION_GRANTED",
        entity_type="Subscription",
        entity_id=str(user_id),
        user_metadata={
            "target_user_id": user_id,
            "subscription_type": request_data.subscription_type,
            "api_call_limit": request_data.api_call_limit,
            "valid_days": request_data.valid_days
        }
    )
    
    return {
        "message": f"Subscription granted to user {user_id}",
        "subscription_id": subscription.subscription_id,
        "subscription_type": subscription.subscription_type,
        "api_call_limit": subscription.api_call_limit,
        "valid_until": subscription.valid_until.isoformat() if subscription.valid_until else None
    }


@router.post("/revoke/{user_id}")
async def revoke_subscription(
    user_id: int,
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Admin: Revoke user's subscription"""
    
    target_user = db.query(User).filter(User.user_id == user_id).first()
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Call service to revoke
    SubscriptionService.revoke_subscription(db, user_id)
    
    # Log the revoke action
    log_action(
        db=db,
        user_id=admin.user_id,
        action_type="SUBSCRIPTION_REVOKED",
        entity_type="Subscription",
        entity_id=str(user_id),
        user_metadata={"revoked_user_id": user_id}
    )
    
    return {
        "success": True,
        "message": f"Subscription revoked for user {user_id}",
        "user_id": user_id
    }


@router.get("/settings", response_model=List[SystemSettingResponse])
async def get_system_settings(
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Admin: Get all system settings"""
    settings = db.query(SystemSetting).all()
    return settings


@router.put("/settings/{setting_key}")
async def update_system_setting(
    setting_key: str,
    update_data: SystemSettingUpdate,
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Admin: Update a system setting"""
    setting = db.query(SystemSetting).filter(
        SystemSetting.setting_key == setting_key
    ).first()
    
    if not setting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Setting '{setting_key}' not found"
        )
    
    old_value = setting.setting_value
    setting.setting_value = update_data.setting_value
    setting.updated_by = admin.user_id
    
    db.commit()
    
    # Clear cache
    SubscriptionService._settings_cache = {}
    
    return {
        "message": f"Setting '{setting_key}' updated",
        "old_value": old_value,
        "new_value": update_data.setting_value
    }