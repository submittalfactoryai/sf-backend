# routers/users.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List
from database import get_db
from models import User, Role, UserRole,AuditLog
from schemas.user import UserUpdateRequest, UserActivationRequest, UserResponse
from core.security import get_current_admin

router = APIRouter(prefix="/api/users", tags=["users"])

@router.get("/", response_model=List[UserResponse])
def list_users(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_admin)   # only admins can access
):
    """
    List all users with their last login time and roles, excluding deleted users.
    Only accessible by admin users.
    """
    last_login_subq = (
        db.query(
            AuditLog.user_id.label("uid"),
            func.max(AuditLog.created_at).label("last_login")
        )
        .filter(AuditLog.action_type == "LOGIN")
        .group_by(AuditLog.user_id)
        .subquery()
    )

    results = (
        db.query(User, last_login_subq.c.last_login)
        .outerjoin(last_login_subq, User.user_id == last_login_subq.c.uid)
        .filter(User.is_deleted == False)      # Only users NOT deleted
        .order_by(User.created_at.desc())
        .all()
    )

    users: List[UserResponse] = []
    for user, last_login in results:
        role_names = [ur.role.name for ur in user.roles]
        can_smart_search = any(r.lower() == "smartsearchuser" or r.lower() == "admin" for r in role_names)
        can_smart_validate = any(r.lower() == "smartvalidateuser" or r.lower() == "admin" for r in role_names)
        users.append(UserResponse(
            id=user.user_id,
            name=user.user_name,
            email=user.email,
            role=role_names[0] if role_names else "user",
            isActive=user.is_active,
            lastLogin=last_login,
            canSmartSearch=can_smart_search,
            canSmartValidate=can_smart_validate,
            roles=role_names
        ))

    return users


@router.patch("/{user_id}/activate", response_model=UserResponse)
def activate_user(
    user_id: int,
    req: UserActivationRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin)
):
    """
    Activate or deactivate a user account (admin only).
    """
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.is_active == req.is_active:
        status_msg = "already active" if req.is_active else "already inactive"
        raise HTTPException(status_code=400, detail=f"User is {status_msg}")
    user.is_active = req.is_active
    db.commit()
    db.refresh(user)


    # Prepare roles (if your UserResponse expects roles as a list)
    role_names = [ur.role.name for ur in user.roles]
    return UserResponse(
        id=user.user_id,
        name=user.user_name,
        email=user.email,
        role=role_names[0] if role_names else "user",
        isActive=user.is_active,
        lastLogin=None,  # Fill if you want to fetch last login
        canSmartSearch=any(r.lower() in ("smartsearchuser", "admin") for r in role_names),
        canSmartValidate=any(r.lower() in ("smartvalidateuser", "admin") for r in role_names),
        roles=role_names
    )


@router.put("/{user_id}/edit", response_model=UserResponse)
def edit_user(
    user_id: int,
    req: UserUpdateRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin)
):
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check for email change conflict
    if req.email and req.email != user.email:
        if db.query(User).filter(User.email == req.email, User.user_id != user_id).first():
            raise HTTPException(status_code=400, detail="Email already exists")
        user.email = req.email
    
    # Update name if provided
    if req.name:
        user.user_name = req.name

    # Get all roles from DB for assignment
    all_roles = db.query(Role).all()
    roles_by_name = {r.name.lower(): r for r in all_roles}

    new_roles = []
    if req.role == "admin":
        # Only admin role assigned
        if "admin" not in roles_by_name:
            raise HTTPException(status_code=400, detail="Admin role not found in roles table")
        new_roles.append(roles_by_name["admin"])
    else:
        # Always assign "user"
        if "user" not in roles_by_name:
            raise HTTPException(status_code=400, detail="User role not found in roles table")
        new_roles.append(roles_by_name["user"])
        if req.canSmartSearch and "smartsearchuser" in roles_by_name:
            new_roles.append(roles_by_name["smartsearchuser"])
        if req.canSmartValidate and "smartvalidateuser" in roles_by_name:
            new_roles.append(roles_by_name["smartvalidateuser"])

    # Clear existing roles and assign new ones
    db.query(UserRole).filter(UserRole.user_id == user_id).delete()
    for role in new_roles:
        db.add(UserRole(user_id=user_id, role_id=role.role_id))

    db.commit()
    db.refresh(user)


    # Compose response
    role_names = [r.name for r in new_roles]
    return UserResponse(
        id=user.user_id,
        name=user.user_name,
        email=user.email,
        role=role_names[0] if role_names else "user",
        isActive=user.is_active,
        lastLogin=None,  # fill with last login if needed
        canSmartSearch=any(r.lower() == "smartsearchuser" or r.lower() == "admin" for r in role_names),
        canSmartValidate=any(r.lower() == "smartvalidateuser" or r.lower() == "admin" for r in role_names),
        roles=role_names
    )


from fastapi.responses import JSONResponse

@router.delete("/{user_id}/delete", status_code=200)
def soft_delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin)
):
    """
    Soft delete a user (admin only): sets is_deleted to True.
    """
    user = db.query(User).filter(User.user_id == user_id, User.is_deleted == False).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found or already deleted")

    user.is_deleted = True
    db.commit()
    db.refresh(user)

    return {"message": "User is deleted", "user_id": user_id}

