# middleware/subscription_middleware.py
# ✅ FIXED: Creates own DB session, proper tracking, reduced logging

from fastapi import Request, HTTPException, status
from sqlalchemy.orm import Session
from services.subscription_service import SubscriptionService
from core.logger import log_action
from models import User
from database import SessionLocal
import time
import logging
import jwt
from config import settings

logger = logging.getLogger(__name__)

# ONLY these endpoints will count towards API limit (PDF uploads)
TRACKED_ENDPOINTS = [
    "/api/extract",
    "/gemini-prod/api/extract"
]

# Endpoints that don't require subscription check at all
EXEMPT_ENDPOINTS = [
    # Health
    "/api/health",
    "/gemini-prod/api/health",
    
    # Auth
    "/api/auth/login",
    "/gemini-prod/api/auth/login",
    "/api/auth/register",
    "/gemini-prod/api/auth/register",
    "/api/auth/authorize",
    "/gemini-prod/api/auth/authorize",
    
    # Subscription endpoints
    "/api/subscription/status",
    "/gemini-prod/api/subscription/status",
    "/api/subscription/user",
    "/gemini-prod/api/subscription/user",
    "/api/subscription/users-with-subscriptions",
    "/gemini-prod/api/subscription/users-with-subscriptions",
    "/api/subscription/grant",
    "/gemini-prod/api/subscription/grant",
    "/api/subscription/revoke",
    "/gemini-prod/api/subscription/revoke",
    "/api/subscription/settings",
    "/gemini-prod/api/subscription/settings",
    
    # Docs
    "/docs",
    "/redoc",
    "/openapi.json",
    
    # Smart Search endpoints (don't count towards limit)
    "/api/search-submittals",
    "/gemini-prod/api/search-submittals",
    "/api/download-pdfs",
    "/gemini-prod/api/download-pdfs",
    "/api/smart-validate-specs",
    "/gemini-prod/api/smart-validate-specs",
    "/api/validate-specs",
    "/gemini-prod/api/validate-specs",
    "/api/proxy-pdf",
    "/gemini-prod/api/proxy-pdf",
    "/api/add-validated-pdfs",
    "/gemini-prod/api/add-validated-pdfs",
    "/api/extract-pds-links",
    "/gemini-prod/api/extract-pds-links",
    "/api/generate-validation-report",
    "/gemini-prod/api/generate-validation-report",
    "/api/download-individual-pdf",
    "/gemini-prod/api/download-individual-pdf",
    "/api/generate-smart-validation-report",
    "/gemini-prod/api/generate-smart-validation-report",
]


async def check_subscription_middleware(request: Request, call_next):
    """
    Middleware to check subscription status ONLY for /api/extract endpoint.
    
    ✅ FIXED: 
    - Always creates own DB session (doesn't depend on db_session_middleware order)
    - Properly increments API call counter for tracked endpoints
    - Commits changes immediately
    - Reduced debug logging
    """
    path = request.url.path
    method = request.method
    
    # Skip check for exempt endpoints (no logging needed)
    if any(path.startswith(exempt) for exempt in EXEMPT_ENDPOINTS):
        return await call_next(request)
    
    # Check if this is a TRACKED endpoint (only /api/extract counts)
    is_tracked_endpoint = any(path.startswith(tracked) for tracked in TRACKED_ENDPOINTS)
    
    if not is_tracked_endpoint:
        # Non-tracked endpoint - proceed without subscription check
        return await call_next(request)
    
    # === TRACKED ENDPOINT: /api/extract ===
    logger.info(f"🛡️ SUBSCRIPTION CHECK: {method} {path}")
    
    # Get token from Authorization header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.warning(f"🛡️ No auth token for tracked endpoint {path}")
        return await call_next(request)
    
    token = auth_header.split(" ")[1]
    
    # Decode JWT to get user_id
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS256"],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
            options={
                "require": ["exp", "iat", "sub", "roles"],
                "verify_exp": True,
                "verify_iat": True,
            },
            leeway=30,
        )
    except jwt.ExpiredSignatureError:
        logger.warning(f"🛡️ Expired token for {path}")
        return await call_next(request)
    except jwt.InvalidTokenError as e:
        logger.warning(f"🛡️ Invalid token for {path}: {e}")
        return await call_next(request)
    
    user_id = int(payload.get("sub"))
    
    # ✅ ALWAYS create our own database session
    # This ensures we don't depend on middleware order
    db = SessionLocal()
    should_close_db = True
    
    try:
        # Load user from database
        user = db.query(User).filter(
            User.user_id == user_id,
            User.is_active == True,
            User.is_deleted == False,
        ).first()
        
        if not user:
            logger.warning(f"🛡️ User {user_id} not found or inactive")
            return await call_next(request)
        
        # Set user in request state for downstream use
        request.state.user = user
        request.state.jwt_payload = payload
        
        # Also set db in state if not already set
        if not hasattr(request.state, 'db') or request.state.db is None:
            request.state.db = db
            should_close_db = False  # Let the main db middleware close it
        
        # === CHECK IF ADMIN (bypass subscription) ===
        role_names = [ur.role.name for ur in user.roles]
        is_admin = any(r.lower() == "admin" for r in role_names)
        
        if is_admin:
            logger.info(f"✅ Admin user {user.user_id} ({user.email}) - bypassing subscription check")
            return await call_next(request)
        
        # === NON-ADMIN USER - CHECK AND UPDATE SUBSCRIPTION ===
        start_time = time.time()
        
        logger.info(f"🔍 Checking subscription for user {user.user_id} ({user.email})")
        
        # Check if user can proceed
        can_proceed, reason = SubscriptionService.check_and_update_subscription_status(db, user.user_id)
        
        if not can_proceed:
            # Log blocked attempt
            log_action(
                db=db,
                user_id=user.user_id,
                action_type="PDF_UPLOAD_BLOCKED",
                entity_type="Subscription",
                entity_id=str(user.user_id),
                user_metadata={
                    "endpoint": path,
                    "method": method,
                    "reason": reason,
                    "action": "blocked"
                },
                process_time=int((time.time() - start_time) * 1000)
            )
            db.commit()
            
            logger.warning(f"❌ PDF upload BLOCKED for user {user.user_id}: {reason}")
            
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "subscription_locked",
                    "message": reason,
                    "contact_admin": SubscriptionService.get_system_setting(
                        db, 'admin_contact_email', 'zack@kbccm.com'
                    )
                }
            )
        
        # ✅ INCREMENT API CALL COUNTER BEFORE processing request
        logger.info(f"📊 Incrementing API call counter for user {user.user_id}")
        increment_success = SubscriptionService.increment_api_call(db, user.user_id)
        
        if increment_success:
            db.commit()  # ✅ CRITICAL: Commit immediately
            logger.info(f"✅ API call incremented for user {user.user_id}")
        else:
            logger.error(f"❌ Failed to increment API call for user {user.user_id}")
        
        # Get updated subscription info AFTER increment
        subscription_info = SubscriptionService.get_subscription_status(db, user.user_id)
        
        # Log successful PDF upload tracking
        log_action(
            db=db,
            user_id=user.user_id,
            action_type="PDF_UPLOAD_TRACKED",
            entity_type="Subscription",
            entity_id=str(user.user_id),
            user_metadata={
                "endpoint": path,
                "method": method,
                "status": "allowed",
                "api_calls_used": subscription_info['api_calls_used'],
                "api_calls_remaining": subscription_info['api_calls_remaining'],
                "subscription_type": subscription_info['subscription_type']
            },
            process_time=int((time.time() - start_time) * 1000)
        )
        db.commit()
        
        logger.info(
            f"✅ PDF upload ALLOWED for user {user.user_id}. "
            f"Usage: {subscription_info['api_calls_used']}/{subscription_info['api_call_limit']} "
            f"({subscription_info['subscription_type']})"
        )
        
        # Proceed with the request
        response = await call_next(request)
        
        # Add subscription info to response headers
        response.headers["X-API-Calls-Used"] = str(subscription_info['api_calls_used'])
        response.headers["X-API-Calls-Remaining"] = str(subscription_info['api_calls_remaining'])
        response.headers["X-Subscription-Type"] = subscription_info['subscription_type']
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions (like 403 Forbidden)
        raise
    except Exception as e:
        logger.error(f"🛡️ Subscription middleware error: {e}", exc_info=True)
        # Don't block the request on middleware errors
        return await call_next(request)
    finally:
        # Close db session if we created it and it wasn't passed to request.state
        if should_close_db and db:
            db.close()