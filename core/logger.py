# core/logger.py
# ✅ FIXED: Handles SSL disconnects WITHOUT creating extra DB connections

from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError, PendingRollbackError, InvalidRequestError
from typing import Optional, Dict, Any
from models import AuditLog
import logging

logger = logging.getLogger(__name__)


def log_action(
    db: Session,
    *,
    user_id: Optional[int],
    action_type: str,
    entity_type: str,
    entity_id: Optional[str] = None,
    user_metadata: Optional[Dict[str, Any]] = None,
    cost_estimate: Optional[float] = None,
    process_time: Optional[int] = None,
    max_retries: int = 2
) -> bool:
    """
    Create an audit log entry in the database with retry logic for SSL disconnects.
    
    This version:
    - Tries with existing session first (retries with rollback)
    - Falls back to fresh session from SAME POOL (no extra connections)
    - Never raises exceptions - returns True/False
    
    Returns:
        bool: True if logging succeeded, False otherwise
    """
    
    # First, try with the provided session
    for attempt in range(max_retries + 1):
        try:
            # Check if session needs rollback
            try:
                if not db.is_active:
                    logger.warning(f"Session not active on attempt {attempt + 1}, rolling back...")
                    db.rollback()
            except Exception:
                pass  # Session might be completely dead
            
            entry = AuditLog(
                user_id=user_id,
                action_type=action_type,
                entity_type=entity_type,
                entity_id=entity_id,
                user_metadata=user_metadata,
                cost_estimate=cost_estimate,
                process_time=process_time
            )
            db.add(entry)
            db.commit()
            return True  # Success
            
        except PendingRollbackError as e:
            logger.warning(f"PendingRollbackError on attempt {attempt + 1}")
            try:
                db.rollback()
            except Exception:
                pass
            if attempt == max_retries:
                break  # Try fresh session
                
        except OperationalError as e:
            error_msg = str(e).lower()
            if "ssl" in error_msg or "closed" in error_msg or "connection" in error_msg:
                logger.warning(f"Connection error on attempt {attempt + 1}: SSL/Connection issue")
            else:
                logger.warning(f"OperationalError on attempt {attempt + 1}: {e}")
            try:
                db.rollback()
            except Exception:
                pass
            if attempt == max_retries:
                break  # Try fresh session
                
        except InvalidRequestError as e:
            logger.warning(f"InvalidRequestError on attempt {attempt + 1}")
            try:
                db.rollback()
            except Exception:
                pass
            if attempt == max_retries:
                break  # Try fresh session
                
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {type(e).__name__}")
            try:
                db.rollback()
            except Exception:
                pass
            if attempt == max_retries:
                break  # Try fresh session
    
    # If all retries with existing session failed, try with a fresh session
    logger.info("Existing session failed, trying with fresh session from pool...")
    return _log_with_fresh_session(
        user_id=user_id,
        action_type=action_type,
        entity_type=entity_type,
        entity_id=entity_id,
        user_metadata=user_metadata,
        cost_estimate=cost_estimate,
        process_time=process_time
    )


def _log_with_fresh_session(
    *,
    user_id: Optional[int],
    action_type: str,
    entity_type: str,
    entity_id: Optional[str] = None,
    user_metadata: Optional[Dict[str, Any]] = None,
    cost_estimate: Optional[float] = None,
    process_time: Optional[int] = None,
) -> bool:
    """
    Log action using a fresh session from the SAME connection pool.
    
    pool_pre_ping=True ensures the connection is validated before use,
    so even if the pool has stale connections, we'll get a working one.
    
    NO EXTRA CONNECTIONS - uses existing pool!
    """
    fresh_session = None
    try:
        # Import here to avoid circular imports
        from database import logging_session_scope
        
        with logging_session_scope() as fresh_session:
            entry = AuditLog(
                user_id=user_id,
                action_type=action_type,
                entity_type=entity_type,
                entity_id=entity_id,
                user_metadata=user_metadata,
                cost_estimate=cost_estimate,
                process_time=process_time
            )
            fresh_session.add(entry)
            # Commit happens automatically in context manager
        
        logger.info("✅ Logged action with fresh session from pool")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to log even with fresh session: {type(e).__name__}: {e}")
        return False


def log_action_safe(
    db: Optional[Session],
    *,
    user_id: Optional[int],
    action_type: str,
    entity_type: str,
    entity_id: Optional[str] = None,
    user_metadata: Optional[Dict[str, Any]] = None,
    cost_estimate: Optional[float] = None,
    process_time: Optional[int] = None,
) -> bool:
    """
    Safe version of log_action that NEVER raises exceptions.
    Always returns True/False to indicate success/failure.
    
    If no db session is provided, uses a fresh session from the pool.
    """
    try:
        if db is not None:
            return log_action(
                db=db,
                user_id=user_id,
                action_type=action_type,
                entity_type=entity_type,
                entity_id=entity_id,
                user_metadata=user_metadata,
                cost_estimate=cost_estimate,
                process_time=process_time
            )
        else:
            return _log_with_fresh_session(
                user_id=user_id,
                action_type=action_type,
                entity_type=entity_type,
                entity_id=entity_id,
                user_metadata=user_metadata,
                cost_estimate=cost_estimate,
                process_time=process_time
            )
    except Exception as e:
        logger.error(f"❌ log_action_safe failed: {type(e).__name__}: {e}")
        return False