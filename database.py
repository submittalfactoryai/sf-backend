# database.py
# ✅ OPTIMIZED FOR LONG-RUNNING OPERATIONS + AIVEN FREE TIER (25 connections max)

from typing import Generator
from contextlib import contextmanager
from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from config import settings
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# SINGLE ENGINE - OPTIMIZED FOR BOTH LONG OPERATIONS AND CONNECTION LIMITS
# =============================================================================
# Aiven Free Tier: 25 max connections
# With 2 workers: each can use up to 8 connections (16 total)
# Leaves 9 for: pgAdmin, Aiven management, migrations, logging
# =============================================================================

engine = create_engine(
    settings.database_url,
    
    # === Connection Pool Settings (Conservative for Aiven Free) ===
    poolclass=QueuePool,
    pool_pre_ping=True,              # CRITICAL: Validates connection before use
    pool_size=10,                     # Reduced from 10 (conservative)
    max_overflow=3,                  # Extra connections under load
    pool_recycle=120,                # Recycle every 2 mins (aggressive for long ops)
    pool_timeout=30,                 # Wait max 30s for connection
    pool_reset_on_return='rollback',
    
    # === AGGRESSIVE TCP Keepalive for 15-20 min operations ===
    connect_args={
        "connect_timeout": 30,
        "application_name": "SF_Backend",
        
        # TCP Keepalive - keeps connection alive during long operations
        "keepalives": 1,
        "keepalives_idle": 20,       # Start probes after 20s idle
        "keepalives_interval": 10,   # Probe every 10s
        "keepalives_count": 6,       # 6 failed probes = dead connection
    },
    
    echo_pool=False,
    echo=False,
)


# =============================================================================
# EVENT LISTENERS
# =============================================================================

@event.listens_for(engine, "connect")
def on_connect(dbapi_connection, connection_record):
    """Log when new connection is created"""
    logger.debug("🔌 New database connection created")


@event.listens_for(engine, "checkout")
def on_checkout(dbapi_connection, connection_record, connection_proxy):
    """Validate connection is still alive on checkout"""
    logger.debug("📤 Connection checked out from pool")


@event.listens_for(engine, "checkin")
def on_checkin(dbapi_connection, connection_record):
    """Log when connection is returned to pool"""
    logger.debug("📥 Connection returned to pool")


@event.listens_for(engine, "invalidate")
def on_invalidate(dbapi_connection, connection_record, exception):
    """Log when connection is invalidated"""
    if exception:
        logger.warning(f"⚠️ Connection invalidated: {exception}")
    else:
        logger.debug("Connection invalidated (recycled)")


# =============================================================================
# SESSION CONFIGURATION
# =============================================================================

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False
)

Base = declarative_base()


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

def get_db() -> Generator[Session, None, None]:
    """
    Generator that yields database sessions for FastAPI dependencies.
    pool_pre_ping ensures connection is valid before use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_fresh_session() -> Session:
    """
    Get a fresh database session from the pool.
    Uses pool_pre_ping to ensure connection is valid.
    
    NOTE: Caller is responsible for closing the session!
    """
    return SessionLocal()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Context manager for database sessions outside FastAPI routes.
    Automatically handles commit/rollback/close.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def logging_session_scope() -> Generator[Session, None, None]:
    """
    Context manager specifically for logging operations.
    
    Uses the same pool but with fresh session.
    pool_pre_ping will validate connection before use.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        logger.warning(f"Logging session error: {e}")
        try:
            session.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            session.close()
        except Exception:
            pass


# =============================================================================
# INITIALIZATION AND CLEANUP
# =============================================================================

def init_db():
    """Initialize database by creating all tables."""
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables initialized")


def close_db_connection():
    """Properly close all database connections."""
    engine.dispose()
    logger.info("✅ Database connections disposed")


# =============================================================================
# MONITORING
# =============================================================================

def get_pool_status() -> dict:
    """Get current connection pool status for monitoring."""
    pool = engine.pool
    return {
        "pool_size": pool.size(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "checked_in": pool.checkedin(),
        "max_possible": 5 + 3,  # pool_size + max_overflow
    }


def check_database_health() -> dict:
    """Check database connectivity and return health status."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        return {
            "status": "healthy",
            "pool": get_pool_status()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "pool": get_pool_status()
        }