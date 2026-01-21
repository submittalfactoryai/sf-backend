# routers/roles.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import get_db
from models import Role
from schemas.role import RoleCreate, RoleResponse
from core.logger import log_action

router = APIRouter(prefix="/api/roles", tags=["roles"])

@router.get("/", response_model=list[RoleResponse])
def list_roles(db: Session = Depends(get_db)):
    roles = db.query(Role).all()
    return [RoleResponse(role_id=r.role_id, name=r.name) for r in roles]

@router.post("/", response_model=RoleResponse, status_code=status.HTTP_201_CREATED)
def create_role(request: RoleCreate, db: Session = Depends(get_db)):
    if db.query(Role).filter(Role.name == request.name).first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Role already exists")
    role = Role(name=request.name)
    db.add(role)
    db.commit()
    db.refresh(role)
    log_action(db, user_id=None, action_type="CREATE_ROLE", entity_type="role", entity_id=str(role.role_id), user_metadata={"name": role.name})
    return RoleResponse(role_id=role.role_id, name=role.name)
