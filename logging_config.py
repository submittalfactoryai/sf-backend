"""
Centralized Logging Configuration for Submittal Factory Backend
Day-wise rotating file logs with structured formatting

Place this file in your backend/ directory alongside api_server.py
"""

import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


def setup_logging(
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    app_name: str = "app"
) -> logging.Logger:
    """
    Setup centralized logging with day-wise rotation.
    
    Args:
        log_dir: Directory to store log files (default: "logs")
        log_level: Logging level (default: INFO)
        app_name: Application name for the logger
    
    Returns:
        Configured root logger
        
    Log files created:
        logs/app.log           - Current day's logs
        logs/app.log.2025-12-09 - Previous day (auto-rotated)
        logs/app.log.2025-12-08 - Day before that
        ... (keeps 30 days)
    """
    # Create logs directory relative to this file's location
    base_dir = Path(__file__).parent.absolute()
    log_path = base_dir / log_dir
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Log file path
    log_file = log_path / f"{app_name}.log"
    
    # Create formatters
    file_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # === File Handler with Day-wise Rotation ===
    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when='midnight',           # Rotate at midnight
        interval=1,                # Every 1 day
        backupCount=30,            # Keep 30 days of logs
        encoding='utf-8',
        delay=False
    )
    file_handler.suffix = '%Y-%m-%d'  # Append date suffix: app.log.2025-12-10
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # === Console Handler ===
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Log startup message
    root_logger.info(f"📁 Logging initialized - Log file: {log_file}")
    root_logger.info(f"📁 Log directory: {log_path}")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger instance.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)