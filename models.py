# models.py
from sqlalchemy import (
    Column, Integer, String, Boolean,
    TIMESTAMP, ForeignKey, JSON, Numeric, BigInteger, func,Text,CheckConstraint
)
from sqlalchemy.orm import relationship
from database import Base

class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    user_name = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=False, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    is_deleted = Column(Boolean, default=False, nullable=False)
    roles = relationship("UserRole", back_populates="user")
    audit_logs = relationship("AuditLog", backref="user")
    subscription_locked = Column(Boolean, default=False)
    last_api_call = Column(TIMESTAMP(timezone=True), nullable=True)
    subscriptions = relationship("UserSubscription", foreign_keys="[UserSubscription.user_id]", back_populates="user")

class Role(Base):
    __tablename__ = "roles"

    role_id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False)

    users = relationship("UserRole", back_populates="role")

class UserRole(Base):
    __tablename__ = "user_roles"

    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True)
    role_id = Column(Integer, ForeignKey("roles.role_id", ondelete="CASCADE"), primary_key=True)

    user = relationship("User", back_populates="roles")
    role = relationship("Role", back_populates="users")

class AuditLog(Base):
    __tablename__ = "audit_logs"

    log_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="SET NULL"), nullable=True)
    action_type = Column(String(50), nullable=False)
    entity_type = Column(String(100), nullable=False)
    entity_id = Column(String(255), nullable=True)
    user_metadata = Column(JSON, nullable=True)
    cost_estimate = Column(Numeric(10,6), nullable=True)
    process_time = Column(BigInteger, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

class UserSubscription(Base):
    __tablename__ = "user_subscriptions"

    subscription_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    subscription_type = Column(String(50), nullable=False, default='free_trial')
    # Types: 'free_trial', 'limited', 'unlimited', 'expired'
    
    api_call_limit = Column(Integer, default=10)
    api_calls_used = Column(Integer, default=0)
    
    valid_from = Column(TIMESTAMP(timezone=True), server_default=func.now())
    valid_until = Column(TIMESTAMP(timezone=True), nullable=True)
    
    is_active = Column(Boolean, default=True)
    granted_by = Column(Integer, ForeignKey("users.user_id", ondelete="SET NULL"), nullable=True)
    
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id], back_populates="subscriptions")
    granter = relationship("User", foreign_keys=[granted_by])
    
    __table_args__ = (
        CheckConstraint('api_calls_used >= 0', name='check_calls_used_positive'),
        CheckConstraint('api_call_limit >= 0 OR api_call_limit = -1', name='check_limit_valid'),
    )


class SystemSetting(Base):
    __tablename__ = "system_settings"

    setting_id = Column(Integer, primary_key=True, index=True)
    setting_key = Column(String(100), unique=True, nullable=False)
    setting_value = Column(Text, nullable=False)
    description = Column(Text)
    updated_by = Column(Integer, ForeignKey("users.user_id", ondelete="SET NULL"))
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    
    updater = relationship("User")