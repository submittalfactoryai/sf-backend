# core/security.py - COMPLETE FIXED VERSION
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from database import get_db
from models import User
import jwt
from jwt import ExpiredSignatureError, InvalidTokenError
import bcrypt
from config import settings
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class TokenPayload(BaseModel):
    sub: str  # user_id
    user_name: str
    email: str
    roles: List[str]
    exp: datetime
    iat: datetime
    iss: Optional[str] = None
    aud: Optional[str] = None


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def get_password_hash(password: str) -> str:
    """
    Generate a secure password hash using bcrypt with increased cost factor
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password string
    """
    salt = bcrypt.gensalt(rounds=14)  # Increased from default 12 for better security
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Securely verify password with constant-time comparison
    
    Args:
        plain_password: Plain text password to verify
        hashed_password: Hashed password from database
        
    Returns:
        True if password matches, False otherwise
    """
    try:
        return bcrypt.checkpw(
            plain_password.encode('utf-8'),
            hashed_password.encode('utf-8')
        )
    except (ValueError, TypeError) as e:
        # Handle invalid hash formats securely
        logger.warning(f"Password verification failed: {str(e)}")
        return False


def create_access_token(
    subject: str,
    user_name: str,
    roles: List[str],
    email: str,
    additional_claims: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a secure JWT token with comprehensive claims
    
    Args:
        subject: User ID (as string)
        user_name: User's display name
        roles: List of user roles
        email: User's email address
        additional_claims: Optional dict of additional JWT claims
        
    Returns:
        Encoded JWT token string
    """
    now = datetime.utcnow()
    expires = now + timedelta(hours=settings.jwt_expiration_hours)
    
    payload = TokenPayload(
        sub=subject,
        user_name=user_name,
        email=email,
        roles=roles,
        exp=expires,
        iat=now,
        iss=settings.jwt_issuer,
        aud=settings.jwt_audience
    ).dict()
    
    # Add any additional claims
    if additional_claims:
        payload.update(additional_claims)
    
    # Generate token with strong algorithm
    token = jwt.encode(
        payload,
        settings.jwt_secret,
        algorithm="HS256",
        headers={
            "typ": "JWT",
            "alg": "HS256",
            "kid": "1"  # Key ID for key rotation scenarios
        }
    )
    
    return token


def verify_token(token: str) -> TokenPayload:
    """
    Verify and decode JWT token with comprehensive validation
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded and validated TokenPayload
        
    Raises:
        HTTPException: If token is invalid or expired
    """
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
                "verify_iat": True
            }
        )
        return TokenPayload(**payload)
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except InvalidTokenError as e:
        logger.warning(f"Invalid token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except jwt.PyJWTError as e:
        logger.error(f"JWT error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token validation failed",
            headers={"WWW-Authenticate": "Bearer"}
        )


async def get_current_user(
    request: Request,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    Get current authenticated user from JWT token
    
    Args:
        request: FastAPI Request object
        token: JWT token from Authorization header
        db: Database session
        
    Returns:
        Authenticated User object
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
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
            leeway=30,  # 30s clock drift tolerance
        )
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except InvalidTokenError as e:
        logger.warning(f"Invalid token in get_current_user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Unexpected error in get_current_user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_id = int(payload.get("sub"))
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user ID in token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = (
        db.query(User)
        .filter(
            User.user_id == user_id,
            User.is_active == True,
            User.is_deleted == False,
        )
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # ✅ IMPROVEMENT: Store user, db, and JWT payload in request state for later use
    request.state.user = user
    request.state.db = db
    request.state.jwt_payload = payload

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Get current active user (alias for get_current_user)
    
    Args:
        current_user: User from get_current_user dependency
        
    Returns:
        Active User object
    """
    return current_user


async def get_current_admin(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current user and verify admin privileges
    
    Args:
        current_user: User from get_current_user dependency
        
    Returns:
        User object with admin privileges
        
    Raises:
        HTTPException: If user doesn't have admin role
    """
    # Check if user has admin role
    user_roles = [ur.role.name.lower() for ur in current_user.roles]
    
    if "admin" not in user_roles:
        logger.warning(
            f"Unauthorized admin access attempt by user {current_user.user_id} "
            f"({current_user.email})"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    
    return current_user


def get_user_roles(user: User) -> List[str]:
    """
    Helper function to extract role names from user object
    
    Args:
        user: User object with roles relationship
        
    Returns:
        List of role names (lowercase)
    """
    return [ur.role.name.lower() for ur in user.roles]


def is_admin(user: User) -> bool:
    """
    Check if user has admin role
    
    Args:
        user: User object
        
    Returns:
        True if user is admin, False otherwise
    """
    return "admin" in get_user_roles(user)


def has_role(user: User, role_name: str) -> bool:
    """
    Check if user has a specific role
    
    Args:
        user: User object
        role_name: Role name to check (case-insensitive)
        
    Returns:
        True if user has the role, False otherwise
    """
    return role_name.lower() in get_user_roles(user)


# ✅ NEW: Rate limiting helper
class RateLimiter:
    """Simple in-memory rate limiter for API endpoints"""
    
    def __init__(self):
        self._attempts = {}
    
    def check_rate_limit(
        self, 
        key: str, 
        max_attempts: int = 5, 
        window_seconds: int = 300
    ) -> bool:
        """
        Check if key has exceeded rate limit
        
        Args:
            key: Unique identifier (e.g., user_id, IP address)
            max_attempts: Maximum attempts allowed in window
            window_seconds: Time window in seconds
            
        Returns:
            True if within limit, False if exceeded
        """
        now = datetime.utcnow()
        
        if key not in self._attempts:
            self._attempts[key] = []
        
        # Remove old attempts outside window
        cutoff = now - timedelta(seconds=window_seconds)
        self._attempts[key] = [
            attempt for attempt in self._attempts[key]
            if attempt > cutoff
        ]
        
        # Check if limit exceeded
        if len(self._attempts[key]) >= max_attempts:
            return False
        
        # Record new attempt
        self._attempts[key].append(now)
        return True


# Global rate limiter instance
rate_limiter = RateLimiter()