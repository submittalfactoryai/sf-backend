# core/logger.py
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
from models import AuditLog

def log_action(
    db: Session,
    *,
    user_id: Optional[int],
    action_type: str,
    entity_type: str,
    entity_id: Optional[str] = None,
    user_metadata: Optional[Dict[str, Any]] = None,
    cost_estimate: Optional[float] = None,
    process_time: Optional[int] = None
) -> None:
    """
    Create an audit log entry in the database.
    """
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
