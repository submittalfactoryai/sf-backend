# database.py

from typing import Generator
from contextlib import contextmanager
from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from config import settings
import logging

logger = logging.getLogger(__name__)

engine = create_engine(
    settings.database_url,
    
    # === Connection Pool Settings ===
    poolclass=QueuePool,
    pool_pre_ping=True,              # Check connection before each use
    pool_size=10,                    # Base connections per worker
    max_overflow=3,                  # Extra connections under load
    pool_recycle=1200,               # Recycle every 20 mins (for long processes)
    pool_timeout=30,                 # Wait max 30s for connection
    pool_reset_on_return='rollback',
    
    # === Connection Settings with TCP Keepalive for Neon DB ===
    connect_args={
        "connect_timeout": 30,
        "application_name": "SF_Backend",
        
        # TCP Keepalive - prevents Neon from killing idle connections
        "keepalives": 1,
        "keepalives_idle": 30,       # Start probes after 30s idle (was 60)
        "keepalives_interval": 10,   # Probe every 10s (was 15)
        "keepalives_count": 6,       # 6 failed probes = dead connection
    },
    
    echo_pool=False,
    echo=False,
)


# =====================================================================
# EVENT LISTENERS (Logging Only - No Extra Queries!)
# =====================================================================

@event.listens_for(engine, "connect")
def on_connect(dbapi_connection, connection_record):
    """Log when new connection is created (no extra queries)"""
    logger.debug("🔌 New database connection created")


@event.listens_for(engine, "checkout")
def on_checkout(dbapi_connection, connection_record, connection_proxy):
    """Log when connection is checked out (no extra queries)"""
    logger.debug("📤 Connection checked out from pool")


@event.listens_for(engine, "checkin")
def on_checkin(dbapi_connection, connection_record):
    """Log when connection is returned to pool"""
    logger.debug("📥 Connection returned to pool")


@event.listens_for(engine, "invalidate")
def on_invalidate(dbapi_connection, connection_record, exception):
    """Log when connection is invalidated (important for debugging)"""
    if exception:
        logger.warning(f"⚠️ Connection invalidated: {exception}")
    else:
        logger.debug("Connection invalidated (recycled)")


# =====================================================================
# SESSION CONFIGURATION
# =====================================================================

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False
)

Base = declarative_base()


# =====================================================================
# SESSION MANAGEMENT
# =====================================================================

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


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Context manager for database sessions outside FastAPI routes.
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


# =====================================================================
# INITIALIZATION AND CLEANUP
# =====================================================================

def init_db():
    """Initialize database by creating all tables."""
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables initialized")


def close_db_connection():
    """Properly close all database connections."""
    engine.dispose()
    logger.info("✅ Database connections disposed")


# =====================================================================
# MONITORING
# =====================================================================

def get_pool_status() -> dict:
    """Get current connection pool status for monitoring."""
    pool = engine.pool
    return {
        "pool_size": pool.size(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "checked_in": pool.checkedin(),
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