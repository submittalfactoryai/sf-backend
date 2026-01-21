from pydantic import BaseModel
from typing import Any

class AuditLogResponse(BaseModel):
    id: int
    userId: int
    user: str
    action: str
    details: Any
    timestamp: str
    cost: float
    process_time: int | None 

    class Config:
        from_attributes = True
