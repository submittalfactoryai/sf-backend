#!/usr/bin/env python3
"""
Standalone backend startup script for Submittal Factory
✅ OPTIMIZED FOR AIVEN FREE TIER
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Start the backend API server"""
    logger.info("Starting Submittal Factory Backend Server")
    logger.info("=" * 60)
    logger.info("⚠️  AIVEN FREE TIER MODE - Limited to 2 workers")
    logger.info("=" * 60)
    
    try:
        # Import configuration
        import config
        settings = config.settings
        logger.info("✅ Configuration loaded successfully")

        # Test DB connection
        from database import engine, get_pool_status
        try:
            with engine.connect():
                logger.info("✅ Database connected successfully")
                pool_status = get_pool_status()
                logger.info(f"📊 Connection Pool: {pool_status}")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            sys.exit(1)

        # Import the FastAPI app
        import api_server
        app = api_server.app
        logger.info("✅ FastAPI app imported successfully")
        
        # Start the server
        import uvicorn
        
        logger.info(f"Starting server on {settings.host}:{settings.port}")
        logger.info("Backend API will be available at:")
        logger.info(f"  - Local: http://localhost:{settings.port}")
        logger.info(f"  - Network: http://{settings.host}:{settings.port}")
        
        # =====================================================================
        # AIVEN FREE TIER: LIMIT TO 2 WORKERS
        # =====================================================================
        # Aiven Free = 25 max connections
        # Each worker creates its own pool: pool_size(5) + max_overflow(3) = 8
        # 2 workers × 8 connections = 16 connections max
        # Leaves 9 connections for: pgAdmin, Aiven management, migrations
        # =====================================================================
        
        # FORCE 2 workers for Aiven Free Tier
        optimal_workers = 2 if not settings.reload else 1
        
        logger.info(f"🔧 Using {optimal_workers} workers (Aiven Free Tier optimized)")
        logger.info(f"📊 Max DB connections: {optimal_workers * 8} (pool_size=5, overflow=3)")
        
        config = uvicorn.Config(
            app=app,
            host=settings.host,
            port=settings.port,
            log_level="info",
            access_log=True,
            reload=settings.reload,
            workers=optimal_workers,
            # Graceful shutdown settings
            timeout_keep_alive=30,
            limit_concurrency=50,  # Max concurrent connections per worker
        )
        
        server = uvicorn.Server(config)
        server.run()
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Make sure you have installed all dependencies:")
        logger.error("  pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start backend server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()