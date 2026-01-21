# services/subscription_service.py
# ✅ FIXED: 
# - Field names match frontend (id, name instead of user_id, user_name)
# - Name field has fallback to email if user_name is null
# - Only locks users, does NOT deactivate (is_active stays True)

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from models import UserSubscription, User, SystemSetting
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SubscriptionService:
    
    # =====================================================================
    # SYSTEM SETTINGS (with simple caching)
    # =====================================================================
    _settings_cache = {}
    _settings_cache_time = None
    _cache_ttl = 300  # 5 minutes
    
    @staticmethod
    def get_system_setting(db: Session, key: str, default: str = None) -> str:
        """Get system setting value with caching"""
        from datetime import datetime
        now = datetime.now()
        
        # Check cache
        cache_key = key
        if (SubscriptionService._settings_cache_time and 
            (now - SubscriptionService._settings_cache_time).seconds < SubscriptionService._cache_ttl and
            cache_key in SubscriptionService._settings_cache):
            return SubscriptionService._settings_cache[cache_key]
        
        # Query database
        setting = db.query(SystemSetting).filter(
            SystemSetting.setting_key == key
        ).first()
        
        value = setting.setting_value if setting else default
        
        # Update cache
        SubscriptionService._settings_cache[cache_key] = value
        SubscriptionService._settings_cache_time = now
        
        return value
    
    
    @staticmethod
    def get_active_subscription(db: Session, user_id: int) -> Optional[UserSubscription]:
        """Get user's active subscription"""
        return db.query(UserSubscription).filter(
            UserSubscription.user_id == user_id,
            UserSubscription.is_active == True
        ).first()
    
    
    @staticmethod
    def create_free_trial(db: Session, user_id: int) -> UserSubscription:
        """Create free trial subscription for new user"""
        # Get default settings
        default_limit = int(SubscriptionService.get_system_setting(
            db, 'default_free_trial_limit', '10'
        ))
        trial_days = int(SubscriptionService.get_system_setting(
            db, 'free_trial_days', '7'
        ))
        
        valid_until = datetime.now(timezone.utc) + timedelta(days=trial_days)
        
        subscription = UserSubscription(
            user_id=user_id,
            subscription_type='free_trial',
            api_call_limit=default_limit,
            api_calls_used=0,
            valid_until=valid_until,
            is_active=True
        )
        
        db.add(subscription)
        db.commit()
        db.refresh(subscription)
        
        logger.info(
            f"Created free trial for user {user_id}: "
            f"{default_limit} PDF uploads allowed, valid until {valid_until} ({trial_days} days)"
        )
        return subscription
    
    
    @staticmethod
    def _calculate_days_remaining(valid_until: Optional[datetime]) -> Optional[int]:
        """
        Calculate days remaining from valid_until date
        Returns: days_remaining as integer, or None if no expiry date
        """
        if not valid_until:
            return None
        
        now = datetime.now(timezone.utc)
        
        # Ensure valid_until is timezone-aware
        if valid_until.tzinfo is None:
            valid_until = valid_until.replace(tzinfo=timezone.utc)
        
        delta = valid_until - now
        days = delta.days
        
        # Return 0 if expired, otherwise return days left
        return max(0, days) if delta.total_seconds() > 0 else 0
    
    
    @staticmethod
    def check_and_update_subscription_status(
        db: Session, 
        user_id: int
    ) -> Tuple[bool, str]:
        """
        Check if user can upload PDFs and update status if needed.
        ✅ FIX: Only locks user, does NOT deactivate (is_active stays True)
        Returns: (can_proceed, reason_message)
        """
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            return False, "User not found"
        
        # Check if user is locked
        if user.subscription_locked:
            admin_email = SubscriptionService.get_system_setting(
                db, 'admin_contact_email', 'zack@kbccm.com'
            )
            return False, (
                f"Your free trial has ended. You cannot upload more PDFs. "
                f"Please contact admin at {admin_email} for continued access."
            )
        
        subscription = SubscriptionService.get_active_subscription(db, user_id)
        
        if not subscription:
            return False, "No active subscription found. Please contact support."
        
        now = datetime.now(timezone.utc)
        
        # Check if subscription expired by date - AUTO LOCK USER (but keep active)
        if subscription.valid_until:
            valid_until = subscription.valid_until
            if valid_until.tzinfo is None:
                valid_until = valid_until.replace(tzinfo=timezone.utc)
            
            if now > valid_until:
                SubscriptionService._lock_subscription_only(db, user, subscription)
                admin_email = SubscriptionService.get_system_setting(
                    db, 'admin_contact_email', 'zack@kbccm.com'
                )
                return False, (
                    f"Your 7-day trial has expired. You cannot upload more PDFs. "
                    f"Please contact admin at {admin_email} for renewal."
                )
        
        # Check unlimited access
        if subscription.subscription_type == 'unlimited' or subscription.api_call_limit == -1:
            return True, "Unlimited access"
        
        # Check PDF upload limit
        calls_remaining = subscription.api_call_limit - subscription.api_calls_used
        
        if calls_remaining <= 0:
            SubscriptionService._lock_subscription_only(db, user, subscription)
            admin_email = SubscriptionService.get_system_setting(
                db, 'admin_contact_email', 'zack@kbccm.com'
            )
            return False, (
                f"You have used all {subscription.api_call_limit} PDF uploads in your free trial. "
                f"Please contact admin at {admin_email} to continue using the service."
            )
        
        return True, f"{calls_remaining} PDF uploads remaining"
    
    
    @staticmethod
    def _lock_subscription_only(db: Session, user: User, subscription: UserSubscription):
        """
        ✅ FIX: Lock subscription ONLY - DO NOT deactivate user (is_active stays True)
        User can still login but cannot upload files
        """
        user.subscription_locked = True
        # ✅ REMOVED: user.is_active = False  (User stays active!)
        subscription.is_active = False
        subscription.subscription_type = 'expired'
        db.commit()
        
        logger.info(
            f"✅ User {user.user_id} subscription locked (user still active and can login). "
            f"Email: {user.email}"
        )
    
    
    @staticmethod
    def increment_api_call(db: Session, user_id: int) -> bool:
        """Increment API call counter for user"""
        subscription = SubscriptionService.get_active_subscription(db, user_id)
        if not subscription:
            logger.warning(f"No active subscription found for user {user_id} when incrementing")
            return False
        
        subscription.api_calls_used += 1
        logger.info(f"Incremented API call for user {user_id}: now {subscription.api_calls_used}")
        return True
    
    
    @staticmethod
    def grant_subscription(
        db: Session, 
        user_id: int, 
        granted_by: int,
        subscription_type: str,
        api_call_limit: Optional[int] = None,
        valid_days: Optional[int] = None
    ) -> UserSubscription:
        """
        Admin grants subscription to user
        ✅ Reactivates user and unlocks subscription
        """
        
        # Find and DELETE ALL existing subscriptions (active and inactive)
        existing_subs = db.query(UserSubscription).filter(
            UserSubscription.user_id == user_id
        ).all()
        
        if existing_subs:
            for sub in existing_subs:
                logger.info(
                    f"Deleting subscription {sub.subscription_id} "
                    f"(active={sub.is_active}, type={sub.subscription_type}) "
                    f"for user {user_id}"
                )
                db.delete(sub)
            
            db.flush()  # Flush deletes before inserting new one
        
        # ✅ Reactivate and unlock user
        user = db.query(User).filter(User.user_id == user_id).first()
        if user:
            user.is_active = True
            user.subscription_locked = False
            logger.info(f"✅ User {user_id} reactivated and unlocked")
        
        # Determine API call limit
        if api_call_limit is None:
            if subscription_type == 'unlimited':
                api_call_limit = -1  # -1 means unlimited
            else:
                api_call_limit = 10  # Default for limited plans
        
        message = f"{subscription_type} subscription"
        if api_call_limit == -1:
            message += " with unlimited PDF uploads"
        else:
            message += f" with {api_call_limit} PDF uploads"
        
        # Calculate validity
        valid_until = None
        if valid_days:
            valid_until = datetime.now(timezone.utc) + timedelta(days=valid_days)
            message += f" for {valid_days} days"
        
        # Create new subscription
        new_subscription = UserSubscription(
            user_id=user_id,
            subscription_type=subscription_type,
            api_call_limit=api_call_limit,
            api_calls_used=0,
            valid_until=valid_until,
            is_active=True,
            granted_by=granted_by
        )
        
        db.add(new_subscription)
        db.commit()
        db.refresh(new_subscription)
        
        logger.info(
            f"✅ Admin {granted_by} granted {subscription_type} subscription to user {user_id}: "
            f"{message}. User reactivated and unlocked."
        )
        
        return new_subscription
    
    
    @staticmethod
    def revoke_subscription(db: Session, user_id: int) -> bool:
        """Revoke user's subscription (lock account)"""
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            return False
        
        # Lock user
        user.subscription_locked = True
        
        # Deactivate subscriptions
        subscription = SubscriptionService.get_active_subscription(db, user_id)
        if subscription:
            subscription.is_active = False
            subscription.subscription_type = 'expired'
        
        db.commit()
        logger.info(f"Revoked subscription for user {user_id}")
        return True
    
    
    @staticmethod
    def get_subscription_status(db: Session, user_id: int) -> dict:
        """
        Get detailed subscription status for user
        ✅ FIX: Check if user is admin FIRST
        """
        user = db.query(User).filter(User.user_id == user_id).first()
        
        if not user:
            admin_email = SubscriptionService.get_system_setting(
                db, 'admin_contact_email', 'zack@kbccm.com'
            )
            return {
                "is_active": False,
                "subscription_type": "none",
                "api_calls_used": 0,
                "api_call_limit": 0,
                "api_calls_remaining": 0,
                "is_expired": True,
                "is_locked": True,
                "admin_contact_email": admin_email,
                "message": "User not found",
                "display_message": f"User not found. Contact {admin_email}",
                "valid_until": None,
                "days_remaining": None
            }
        
        # ✅ CHECK IF USER IS ADMIN FIRST
        role_names = [ur.role.name for ur in user.roles]
        is_admin = any(r.lower() == "admin" for r in role_names)
        
        if is_admin:
            # Admin users always have unlimited access
            return {
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
        
        # Continue with regular subscription logic for non-admin users
        subscription = SubscriptionService.get_active_subscription(db, user_id)
        admin_email = SubscriptionService.get_system_setting(
            db, 'admin_contact_email', 'zack@kbccm.com'
        )
        
        if not subscription:
            return {
                "is_active": False,
                "subscription_type": "none",
                "api_calls_used": 0,
                "api_call_limit": 0,
                "api_calls_remaining": 0,
                "is_expired": True,
                "is_locked": user.subscription_locked if user else True,
                "admin_contact_email": admin_email,
                "message": "No active subscription. Contact admin to activate.",
                "display_message": f"No active subscription. Contact {admin_email}",
                "valid_until": None,
                "days_remaining": None
            }
        
        # Calculate days remaining
        days_remaining = SubscriptionService._calculate_days_remaining(subscription.valid_until)
        
        is_expired = False
        if subscription.valid_until:
            valid_until = subscription.valid_until
            if valid_until.tzinfo is None:
                valid_until = valid_until.replace(tzinfo=timezone.utc)
            is_expired = datetime.now(timezone.utc) > valid_until
        
        remaining = -1  # Unlimited
        if subscription.api_call_limit >= 0:
            remaining = max(0, subscription.api_call_limit - subscription.api_calls_used)
        
        # Create user-friendly message
        if user.subscription_locked or is_expired:
            display_message = f"Trial ended. Contact {admin_email} to continue."
        elif subscription.subscription_type == 'unlimited' or remaining == -1:
            display_message = "Unlimited PDF uploads available"
        elif remaining == 0:
            display_message = f"No uploads remaining. Contact {admin_email}"
        elif remaining <= 2:
            display_message = f"⚠️ Only {remaining} PDF upload(s) remaining"
        else:
            display_message = f"{remaining} PDF uploads remaining"
        
        return {
            "is_active": subscription.is_active and not is_expired,
            "subscription_type": subscription.subscription_type,
            "api_calls_used": subscription.api_calls_used,
            "api_call_limit": subscription.api_call_limit,
            "api_calls_remaining": remaining,
            "is_expired": is_expired,
            "is_locked": user.subscription_locked if user else False,
            "admin_contact_email": admin_email,
            "valid_until": subscription.valid_until.isoformat() if subscription.valid_until else None,
            "display_message": display_message,
            "days_remaining": days_remaining
        }
    
    
    @staticmethod
    def get_all_users_with_subscriptions(
        db: Session,
        skip: int = 0,
        limit: int = 50
    ) -> dict:
        """
        Get all users with their subscription data in ONE query.
        ✅ FIXED: Uses "id" and "name" to match frontend interface.
        ✅ FIXED: Name field has fallback to email if user_name is null.
        """
        from models import User, UserSubscription
        
        # Get total count
        total = db.query(User).filter(User.is_deleted == False).count()
        
        # Get users with pagination
        users = db.query(User).filter(
            User.is_deleted == False
        ).offset(skip).limit(limit).all()
        
        admin_email = SubscriptionService.get_system_setting(
            db, 'admin_contact_email', 'zack@kbccm.com'
        )
        
        result_users = []
        now = datetime.now(timezone.utc)
        
        for user in users:
            # Check if user is admin
            role_names = [ur.role.name for ur in user.roles]
            is_admin = any(r.lower() == "admin" for r in role_names)
            
            # ✅ FIXED: Build user data with correct field names for frontend
            user_data = {
                "id": user.user_id,                                    # Frontend expects "id"
                "name": user.user_name or user.email or "Unknown",     # Frontend expects "name" with fallback
                "email": user.email or "",
                "is_active": user.is_active,
                "subscription": None
            }
            
            if is_admin:
                # Admin users have special subscription
                user_data["subscription"] = {
                    "user_id": user.user_id,
                    "subscription_type": "admin",
                    "is_active": True,
                    "api_calls_used": 0,
                    "api_call_limit": -1,
                    "api_calls_remaining": -1,
                    "is_expired": False,
                    "is_locked": False,
                    "valid_until": None,
                    "expiry_date": None,
                    "start_date": None,
                    "days_remaining": None,
                    "admin_contact_email": ""
                }
            else:
                # Get active subscription for non-admin users
                subscription = db.query(UserSubscription).filter(
                    UserSubscription.user_id == user.user_id,
                    UserSubscription.is_active == True
                ).first()
                
                if subscription:
                    # Calculate days remaining
                    days_remaining = SubscriptionService._calculate_days_remaining(subscription.valid_until)
                    
                    # Check if expired
                    is_expired = False
                    if subscription.valid_until:
                        valid_until = subscription.valid_until
                        if valid_until.tzinfo is None:
                            valid_until = valid_until.replace(tzinfo=timezone.utc)
                        is_expired = now > valid_until
                    
                    # Calculate remaining calls
                    remaining = -1  # Unlimited
                    if subscription.api_call_limit >= 0:
                        remaining = max(0, subscription.api_call_limit - subscription.api_calls_used)
                    
                    user_data["subscription"] = {
                        "user_id": subscription.user_id,
                        "subscription_type": subscription.subscription_type,
                        "is_active": subscription.is_active and not is_expired,
                        "api_calls_used": subscription.api_calls_used,
                        "api_call_limit": subscription.api_call_limit,
                        "api_calls_remaining": remaining,
                        "is_expired": is_expired,
                        "is_locked": user.subscription_locked,
                        "valid_until": subscription.valid_until.isoformat() if subscription.valid_until else None,
                        "expiry_date": subscription.valid_until.isoformat() if subscription.valid_until else None,
                        "start_date": subscription.valid_from.isoformat() if subscription.valid_from else None,
                        "days_remaining": days_remaining if days_remaining is not None else 0,
                        "admin_contact_email": admin_email
                    }
            
            result_users.append(user_data)
        
        return {
            "users": result_users,
            "total": total,
            "page": skip // limit if limit > 0 else 0,
            "limit": limit,
            "has_more": (skip + limit) < total
        }
    
    
    @staticmethod
    def check_expired_trials_bulk(db: Session) -> int:
        """
        Bulk check and lock all users with expired trials.
        ✅ FIX: Only locks, does NOT deactivate users.
        Should be called by a background job/cron.
        Returns: Number of users locked
        """
        now = datetime.now(timezone.utc)
        
        # Find all active subscriptions that are expired
        expired_subscriptions = db.query(UserSubscription).filter(
            UserSubscription.is_active == True,
            UserSubscription.valid_until <= now
        ).all()
        
        locked_count = 0
        for subscription in expired_subscriptions:
            user = db.query(User).filter(User.user_id == subscription.user_id).first()
            if user and not user.subscription_locked:
                SubscriptionService._lock_subscription_only(db, user, subscription)
                locked_count += 1
        
        logger.info(f"Bulk trial check: {locked_count} users locked (still active)")
        return locked_count