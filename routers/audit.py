from datetime import datetime, timedelta
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import get_db
from models import AuditLog, User
from schemas.audit import AuditLogResponse
from typing import List

router = APIRouter(prefix="/api/audit", tags=["audit"])

@router.get("/", response_model=List[AuditLogResponse])
def get_audit_logs(db: Session = Depends(get_db)):
    """
    Return audit log entries from the last 5 days, newest first.
    """
    # calculate cutoff timestamp
    five_days_ago = datetime.utcnow() - timedelta(days=5)

    # join with User, filter by created_at >= five_days_ago
    logs = (
        db.query(AuditLog, User)
        .join(User, AuditLog.user_id == User.user_id)
        .filter(AuditLog.created_at >= five_days_ago)
        .order_by(AuditLog.created_at.desc())
        .all()
    )

    return [
        AuditLogResponse(
            id=audit.log_id,
            userId=audit.user_id,
            user=user.user_name,
            action=audit.action_type.replace("_", " ").title(),
            details=audit.user_metadata,
            timestamp=audit.created_at.isoformat(),
            cost=float(audit.cost_estimate) if audit.cost_estimate is not None else 0.00,
            process_time=audit.process_time,
        )
        for audit, user in logs
    ]
