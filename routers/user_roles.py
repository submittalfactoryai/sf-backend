# routers/user_roles.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import get_db
from models import User, Role, UserRole
from schemas.role import RoleResponse, AssignRolesRequest
from core.logger import log_action

router = APIRouter(prefix="/api/users", tags=["user_roles"])

@router.post("/{user_id}/roles", response_model=list[RoleResponse])
def assign_roles(
    user_id: int,
    request: AssignRolesRequest,
    db: Session = Depends(get_db)
) -> list[RoleResponse]:
    # 1. Ensure user exists
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # 2. Validate requested roles
    roles = db.query(Role).filter(Role.role_id.in_(request.role_ids)).all()
    if len(roles) != len(request.role_ids):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="One or more roles not found")

    # 3. Clear existing roles
    db.query(UserRole).filter(UserRole.user_id == user_id).delete()

    # 4. Assign new roles
    for role in roles:
        db.add(UserRole(user_id=user_id, role_id=role.role_id))
    db.commit()

    # 5. Audit log
    log_action(db, user_id=user_id, action_type="ASSIGN_ROLES", entity_type="user", entity_id=str(user_id), user_metadata={"roles": [r.name for r in roles]})

    return [RoleResponse(role_id=r.role_id, name=r.name) for r in roles]