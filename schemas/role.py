# schemas/role.py
from pydantic import BaseModel, constr

class RoleCreate(BaseModel):
    name: constr(min_length=1, max_length=50)

class RoleResponse(BaseModel):
    role_id: int
    name: constr(min_length=1, max_length=50)

class AssignRolesRequest(BaseModel):
    role_ids: list[int]
