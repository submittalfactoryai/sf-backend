# routers/auth.py - FIXED VERSION
# ✅ FIX: Allow login for subscription-locked users, only show locked state

import time
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from core.security import get_password_hash, verify_password, create_access_token
from core.logger import log_action
from database import get_db
from models import User, Role, UserRole
from core.security import get_current_active_user, get_current_user
from schemas.auth import RegisterRequest, RegisterResponse, LoginRequest, LoginResponse, AdminPasswordResetRequest, PasswordResetResponse
from services.subscription_service import SubscriptionService

router = APIRouter(prefix="/api/auth", tags=["auth"])

class AuthorizeResponse(BaseModel):
    user_id: int
    email: str
    user_name: str
    roles: list[str] = Field(default_factory=list)
    iat: int
    exp: int
    now: int
    seconds_until_expiry: int


# ============================
# REGISTER ENDPOINT
# ============================
@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED
)
async def register(
    request: Request,
    register_data: RegisterRequest,
    db: Session = Depends(get_db)
) -> RegisterResponse:
    """Register a new user with automatic activation"""
    start_time = time.time()
    try:
        # Check if email already exists (including deleted users)
        existing_user = db.query(User).filter(User.email == register_data.email).first()
        
        if existing_user:
            if existing_user.is_deleted:
                # Email exists but user is soft-deleted - RESTORE the account
                print(f"🔄 Restoring soft-deleted account for: {register_data.email}")
                
                # Update user details
                existing_user.user_name = register_data.user_name
                existing_user.password_hash = get_password_hash(register_data.password)
                existing_user.is_deleted = False  # RESTORE user
                existing_user.is_active = True    # ACTIVATE user
                existing_user.subscription_locked = False
                existing_user.updated_at = datetime.utcnow()
                
                db.commit()
                db.refresh(existing_user)
                
                # Ensure default roles are assigned
                existing_roles = db.query(UserRole).filter(UserRole.user_id == existing_user.user_id).all()
                if not existing_roles:
                    default_user_role = db.query(Role).filter(Role.role_id == 2).first()  # user role
                    smart_search_role = db.query(Role).filter(Role.role_id == 3).first()  # smartsearchuser role
                    
                    if default_user_role:
                        db.add(UserRole(user_id=existing_user.user_id, role_id=default_user_role.role_id))
                    
                    if smart_search_role:
                        db.add(UserRole(user_id=existing_user.user_id, role_id=smart_search_role.role_id))
                    
                    db.commit()
                
                # Create/restore subscription
                existing_subscription = SubscriptionService.get_active_subscription(db, existing_user.user_id)
                if not existing_subscription:
                    subscription = SubscriptionService.create_free_trial(db, existing_user.user_id)
                else:
                    existing_subscription.is_active = True
                    db.commit()
                
                process_time = int((time.time() - start_time) * 1000)
                
                # Audit log for restoration
                log_action(
                    db=db,
                    user_id=existing_user.user_id,
                    action_type="REGISTER_RESTORED",
                    entity_type="User",
                    entity_id=str(existing_user.user_id),
                    user_metadata={
                        "email": existing_user.email,
                        "username": existing_user.user_name,
                        "status": "restored",
                        "previous_status": "deleted",
                        "roles": ["user", "smartsearchuser"]
                    },
                    process_time=process_time
                )
                
                return RegisterResponse(
                    user_id=existing_user.user_id,
                    email=existing_user.email,
                    user_name=existing_user.user_name
                )
            else:
                # Email exists and user is NOT deleted - throw error
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )

        # NORMAL FLOW: No existing user found - create new user
        hashed_pwd = get_password_hash(register_data.password)
        user = User(
            email=register_data.email,
            user_name=register_data.user_name,
            password_hash=hashed_pwd,
            is_active=True,
            is_deleted=False,
            subscription_locked=False
        )
        
        db.add(user)
        db.commit()
        db.refresh(user)

        # Assign default roles - 'user' (id=2) AND 'smartsearchuser' (id=3)
        default_user_role = db.query(Role).filter(Role.role_id == 2).first()  # user role
        smart_search_role = db.query(Role).filter(Role.role_id == 3).first()  # smartsearchuser role
        
        if default_user_role:
            db.add(UserRole(user_id=user.user_id, role_id=default_user_role.role_id))
        
        if smart_search_role:
            db.add(UserRole(user_id=user.user_id, role_id=smart_search_role.role_id))
            
        db.commit()
        
        # Create free trial subscription (7 days, 10 API calls)
        subscription = SubscriptionService.create_free_trial(db, user.user_id)

        # Calculate process time
        process_time = int((time.time() - start_time) * 1000)

        # Enhanced audit log
        assigned_roles = []
        if default_user_role:
            assigned_roles.append("user")
        if smart_search_role:
            assigned_roles.append("smartsearchuser")
            
        log_action(
            db=db,
            user_id=user.user_id,
            action_type="REGISTER",
            entity_type="User",
            entity_id=str(user.user_id),
            user_metadata={
                "email": user.email,
                "username": user.user_name,
                "status": "active",
                "roles": assigned_roles,
                "subscription_type": "free_trial",
                "api_call_limit": subscription.api_call_limit,
                "trial_days": 7
            },
            process_time=process_time
        )

        return RegisterResponse(
            user_id=user.user_id,
            email=user.email,
            user_name=user.user_name
        )

    except HTTPException as http_exc:
        process_time = int((time.time() - start_time) * 1000)
        log_action(
            db=db,
            user_id=None,
            action_type="REGISTER_FAILED",
            entity_type="User",
            user_metadata={
                "email": register_data.email,
                "error": http_exc.detail,
                "status": "failed"
            },
            process_time=process_time
        )
        raise http_exc

    except Exception as e:
        process_time = int((time.time() - start_time) * 1000)
        log_action(
            db=db,
            user_id=None,
            action_type="REGISTER_FAILED",
            entity_type="User",
            user_metadata={
                "email": register_data.email,
                "error": str(e),
                "status": "failed"
            },
            process_time=process_time
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed due to internal error"
        )


# ============================
# LOGIN ENDPOINT - FIXED
# ============================
@router.post(
    "/login",
    response_model=LoginResponse,
    status_code=status.HTTP_200_OK
)
async def login(
    request: Request,
    login_data: LoginRequest,
    db: Session = Depends(get_db)
) -> LoginResponse:
    """
    User login endpoint
    ✅ FIX: Allow login even if subscription is locked
    User can still login but will see subscription-locked UI
    """
    start_time = time.time()
    
    try:
        # Find user (exclude deleted users)
        user = db.query(User).filter(
            User.email == login_data.email,
            User.is_deleted == False
        ).first()
        
        # Verify password
        if not user or not verify_password(login_data.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # ✅ FIX: Check if user is admin - admins bypass all checks
        role_names = [ur.role.name for ur in user.roles]
        is_admin = any(r.lower() == "admin" for r in role_names)
        
        # ✅ FIX: Only block login if user is explicitly deactivated (not just subscription locked)
        if not user.is_active and not is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is inactive"
            )
        
        # Calculate permissions
        can_smart_search = any(r.lower() in ["smartsearchuser", "admin"] for r in role_names)
        can_smart_validate = any(r.lower() in ["smartvalidateuser", "admin"] for r in role_names)

        # Get subscription status
        if is_admin:
            subscription_status = {
                "is_active": True,
                "subscription_type": "admin",
                "api_calls_used": 0,
                "api_call_limit": -1,
                "api_calls_remaining": -1,
                "is_expired": False,
                "is_locked": False,
                "admin_contact_email": "",
                "valid_until": None,
                "display_message": "Admin account - unlimited access",
                "days_remaining": None
            }
        else:
            # Get subscription status for non-admin users
            subscription_status = SubscriptionService.get_subscription_status(db, user.user_id)
        

        # Create access token
        token = create_access_token(
            subject=str(user.user_id),
            user_name=user.user_name,
            roles=role_names,
            email=user.email
        )

        # Calculate process time
        process_time = int((time.time() - start_time) * 1000)

        # Enhanced audit log
        log_action(
            db=db,
            user_id=user.user_id,
            action_type="LOGIN",
            entity_type="User",
            entity_id=str(user.user_id),
            user_metadata={
                "email": user.email,
                "ip_address": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
                "roles": role_names,
                "auth_method": "password",
                "is_admin": is_admin,
                "subscription_locked": user.subscription_locked
            },
            process_time=process_time
        )

        return LoginResponse(
            access_token=token,
            token_type="bearer",
            user_id=user.user_id,
            user_name=user.user_name,
            email=user.email,
            roles=role_names,
            isActive=user.is_active,
            canSmartSearch=can_smart_search,
            canSmartValidate=can_smart_validate,
            subscription=subscription_status
        )

    except HTTPException as http_exc:
        process_time = int((time.time() - start_time) * 1000)
        log_action(
            db=db,
            user_id=None,
            action_type="LOGIN_FAILED",
            entity_type="User",
            user_metadata={
                "email": login_data.email,
                "ip_address": request.client.host if request.client else None,
                "error": http_exc.detail,
                "status": "failed"
            },
            process_time=process_time
        )
        raise http_exc

    except Exception as e:
        process_time = int((time.time() - start_time) * 1000)
        log_action(
            db=db,
            user_id=None,
            action_type="LOGIN_FAILED",
            entity_type="User",
            user_metadata={
                "email": login_data.email,
                "ip_address": request.client.host if request.client else None,
                "error": str(e),
                "status": "failed"
            },
            process_time=process_time
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed due to internal error"
        )


# ============================
# OTHER ENDPOINTS (UNCHANGED)
# ============================
@router.post(
    "/user/reset-password",
    response_model=PasswordResetResponse,
    status_code=status.HTTP_200_OK,
    tags=["admin"]
)
async def admin_reset_password(
    request: Request,
    body: AdminPasswordResetRequest,
    db: Session = Depends(get_db),
    admin_user = Depends(get_current_active_user),
):
    """Reset a user's password. Only accessible by admin."""
    start_time = time.time()
    # Find user (exclude deleted users)
    user = db.query(User).filter(
        User.user_id == body.user_id,
        User.is_deleted == False
    ).first()
    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    # Update password
    user.password_hash = get_password_hash(body.new_password)
    db.commit()

    process_time = int((time.time() - start_time) * 1000)
    # Audit log
    log_action(
        db=db,
        user_id=admin_user.user_id,
        action_type="USER_PASSWORD_RESET",
        entity_type="User",
        entity_id=str(user.user_id),
        user_metadata={
            "reset_by_admin": admin_user.email,
            "target_user": user.email,
            "target_user_id": user.user_id
        },
        process_time=process_time
    )
    return PasswordResetResponse(
        user_id=user.user_id,
        email=user.email,
        user_name=user.user_name,
        message="Password has been reset successfully."
    )


@router.get(
    "/authorize",
    response_model=AuthorizeResponse,
    status_code=status.HTTP_200_OK,
)
async def authorize(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> AuthorizeResponse:
    payload = getattr(request.state, "jwt_payload", None) or {}
    exp = int(payload.get("exp"))
    iat = int(payload.get("iat"))
    now = int(datetime.now(timezone.utc).timestamp())

    return AuthorizeResponse(
        user_id=current_user.user_id,
        email=current_user.email,
        user_name=current_user.user_name,
        roles=[ur.role.name for ur in current_user.roles],
        iat=iat,
        exp=exp,
        now=now,
        seconds_until_expiry=max(0, exp - now),
    )


@router.head("/authorize", status_code=status.HTTP_200_OK)
async def authorize_head(
    current_user: User = Depends(get_current_user),
):
    return