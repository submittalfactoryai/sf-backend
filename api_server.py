# type: ignore
# pyright: reportUnusedVariable=false
# pyright: reportUnusedImport=false
"""
Submittal Factory API Server
This file has Pylance warnings suppressed due to dynamic imports.
The code is fully functional.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Query, Form, Depends, Header, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from database import init_db, close_db_connection
from slowapi.errors import RateLimitExceeded
import os
import base64
import re
import shutil
import tempfile
from pathlib import Path
# Import refactored functions
try:
    from extract_with_llm import extract_product_data, MODEL_NAME as LLM_MODEL_NAME
    from searching_pdf import search_pdfs_serpapi
except ImportError:
    from .extract_with_llm import extract_product_data, MODEL_NAME as LLM_MODEL_NAME
    from .searching_pdf import search_pdfs_serpapi
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import io
import zipfile
import logging
import traceback
# Import refactored functions and constants
try:
    from to_text import extract_text_between_patterns, check_part2_exists, PART_2_START_REGEX, PART_3_START_REGEX
    from clean_text import clean_text
    from to_json import parse_text_to_json_structure
    from metadata import generate_metadata as generate_metadata_and_get_tokens, MODEL_NAME as METADATA_MODEL_NAME

    from PDF_link_extraction import extract_pdf_links

    # Import lightning-fast preprocessing
    try:
        from lightning_fast_preprocessing import (
            lightning_fast_preprocessing_pipeline,
            lightning_fast_prepare_product_data,
            lightning_fast_preprocessing_pipeline_with_patterns,  # ✅ ADDED
            get_preprocessing_stats
        )
    except ImportError:
        from .lightning_fast_preprocessing import (
            lightning_fast_preprocessing_pipeline,
            lightning_fast_prepare_product_data,
            lightning_fast_preprocessing_pipeline_with_patterns,  # ✅ ADDED
            get_preprocessing_stats
        )
except ImportError:
    from .to_text import extract_text_between_patterns, check_part2_exists, PART_2_START_REGEX, PART_3_START_REGEX
    from .clean_text import clean_text
    from .to_json import parse_text_to_json_structure
    from .metadata import generate_metadata as generate_metadata_and_get_tokens, MODEL_NAME as METADATA_MODEL_NAME
    from .PDF_link_extraction import extract_pdf_links
    # Import lightning-fast preprocessing
    from .lightning_fast_preprocessing import (
        lightning_fast_preprocessing_pipeline,
        lightning_fast_prepare_product_data,
        ightning_fast_preprocessing_pipeline_with_patterns,
        get_preprocessing_stats
    )
# --- Keep existing imports ---
from pypdf import PdfReader, PdfWriter
try:
    from pdf_view_markers import router as pdf_markers_router
    from trim_pdf_section import router as trim_pdf_router
    from pdf_report_generator import generate_validation_report_and_merge
except ImportError:
    from .pdf_view_markers import router as pdf_markers_router
    from .trim_pdf_section import router as trim_pdf_router
    from .pdf_report_generator import generate_validation_report_and_merge
import csv
import datetime
import threading
import httpx
from urllib.parse import urlparse
from contextlib import asynccontextmanager
import aiofiles
from fastapi.concurrency import run_in_threadpool
import asyncio
import sys
import hashlib
from collections import defaultdict
import time

# Import configuration
try:
    from config import settings
except ImportError:
    # Fallback for direct execution
    from .config import settings

# near the top of api_server.py, after creating `app = FastAPI(...)`
from routers.auth import router as auth_router

from routers.roles import router as roles_router

from routers.user_roles import router as user_roles_router

from routers.users import router as users_router

from routers.audit import router as audit_router

from core.security import get_current_active_user

from routers.pds_add_more import router as pds_add_more_router

from database import get_db
from sqlalchemy.orm import Session


from core.logger import log_action
from models import AuditLog
from models import User 
from schemas.user import CustomAuditLogRequest

# Helper function for audit logging
from decimal import Decimal
from typing import Optional, Dict, Any

from middleware.subscription_middleware import check_subscription_middleware
from routers.subscription import router as subscription_router

from services.background_tasks import start_background_scheduler
import atexit

try:
    from uploadServices.section_detection_service import (
        detect_products_section,
        get_section_extraction_patterns,  # ✅ ADDED - needed for section-aware extraction
        # Import custom exceptions for proper error handling
        SectionDetectionError,
        GeminiNotAvailableError,
        APIKeyNotConfiguredError,
        ModelNotFoundError,
        GeminiAPIError,
        PDFExtractionError,
        InvalidResponseError
    )
    from uploadServices.extraction_warnings import (
        PRODUCT_COUNT_WARNING_THRESHOLD,
        PRODUCT_COUNT_CRITICAL_THRESHOLD
    )
    SECTION_DETECTION_AVAILABLE = True
except ImportError:
    SECTION_DETECTION_AVAILABLE = False
    PRODUCT_COUNT_WARNING_THRESHOLD = 100
    PRODUCT_COUNT_CRITICAL_THRESHOLD = 200
    # Define placeholder exceptions if import fails
    class SectionDetectionError(Exception): pass
    class GeminiNotAvailableError(SectionDetectionError): pass
    class APIKeyNotConfiguredError(SectionDetectionError): pass
    class ModelNotFoundError(SectionDetectionError): pass
    class GeminiAPIError(SectionDetectionError): pass
    class PDFExtractionError(SectionDetectionError): pass
    class InvalidResponseError(SectionDetectionError): pass

# Add after existing imports - Error handling system
# NOTE: Change 'utils.error_handlers' to 'error_handlers' if error_handlers.py is in backend/ root
#       Or keep 'utils.error_handlers' if error_handlers.py is in backend/utils/
try:
    from error_handlers import (
        ErrorCode,
        ErrorCategory,
        create_error_response,
        classify_llm_error,
        classify_network_error,
        classify_extraction_error,
        SubmittalFactoryException,
        handle_gemini_exception,
        upload_error,
        extraction_error,
        validation_error,
        search_error,
        ERROR_MESSAGES,  # Added - this was missing and causing the error
    )
except ImportError:
    from utils.error_handlers import (
        ErrorCode,
        ErrorCategory,
        create_error_response,
        classify_llm_error,
        classify_network_error,
        classify_extraction_error,
        SubmittalFactoryException,
        handle_gemini_exception,
        upload_error,
        extraction_error,
        validation_error,
        search_error,
        ERROR_MESSAGES,  # Added - this was missing and causing the error
    )

from logging_config import setup_logging

async def log_api_success(
    request: Request,
    action: str,
    entity_type: str,
    entity_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    cost: Optional[float] = None,
    start_time: Optional[float] = None
):
    """Helper for successful API calls - with error protection"""
    try:
        process_time = int((time.time() - start_time) * 1000) if start_time else None
        cost_decimal = Decimal(str(cost)).quantize(Decimal('0.000001')) if cost is not None else None
        
        db = getattr(request.state, 'db', None)
        if db is None:
            logger.warning("No database session available for logging")
            return
            
        log_action(
            db=db,
            user_id=getattr(request.state, 'user', None) and request.state.user.user_id,
            action_type=action,
            entity_type=entity_type,
            entity_id=entity_id,
            user_metadata=metadata or {},
            cost_estimate=cost_decimal,
            process_time=process_time
        )
    except Exception as e:
        logger.error(f"Failed to log API success: {e}")
        # Don't re-raise - logging failure shouldn't break the API response



async def log_api_failure(
    request: Request,
    action: str,
    entity_type: str,
    entity_id: Optional[str] = None,
    error: Optional[str] = None,
    start_time: Optional[float] = None
):
    """Helper for failed API calls - with error protection"""
    try:
        process_time = int((time.time() - start_time) * 1000) if start_time else None
        
        db = getattr(request.state, 'db', None)
        if db is None:
            logger.warning("No database session available for logging")
            return
        
        # Try to get user_id safely without triggering lazy load
        user_id = None
        try:
            user = getattr(request.state, 'user', None)
            if user is not None:
                # Access the ID directly if already loaded
                user_id = user.__dict__.get('user_id')
        except Exception:
            pass  # Can't get user_id, that's okay
        
        log_action(
            db=db,
            user_id=user_id,
            action_type=f"{action}_Failed",
            entity_type=entity_type,
            entity_id=entity_id,
            user_metadata={"error": str(error)[:500]} if error else {},
            process_time=process_time
        )
    except Exception as e:
        logger.error(f"Failed to log API failure: {e}")
        # Don't re-raise - logging failure shouldn't break the error response

# Initialize logging ONCE at module load
# Creates: logs/app.log (current), logs/app.log.2025-12-09 (rotated), etc.
setup_logging(
    log_dir="logs",        # Folder for log files
    log_level=logging.INFO,
    app_name="app"         # Base filename: app.log
)
logger = logging.getLogger(__name__)

from database import engine

try:
    with engine.connect():
        logger.info("✅ Database connected successfully")
except Exception as e:
    logger.error(f"❌ Database connection failed: {e}")
    sys.exit(1)

# Configure basic logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Reduce verbosity for less important logs
# logger.setLevel(logging.INFO)

# --- Pricing Constants (USD per 1 Million Tokens) ---
MODEL_PRICING = {
    "gemini-2.0-flash": {
        "input": 0.1,
        "output": 0.4
    },
    "gemini-1.5-flash": {
        "input": 0.075,
        "output": 0.3
    },
    "gemini-2.5-flash-preview-04-17": {
        "input": 0.15,
        "output": 0.6
    },
    "unknown_model": {
        "input": 0.15,
        "output": 0.60
    }
}
GROUNDED_SEARCH_FIXED_COST_PER_REQUEST = 0.035 # $35 per 1000 requests = $0.035 per request

# --- Token Cost Calculation Function ---
def get_token_cost(model_name, prompt_tokens, completion_tokens):
    """Calculate the cost in USD for token usage of a specific model."""
    pricing = MODEL_PRICING.get(model_name, MODEL_PRICING["unknown_model"])
    input_cost = (prompt_tokens * pricing["input"]) / 1_000_000
    output_cost = (completion_tokens * pricing["output"]) / 1_000_000
    return input_cost + output_cost

# --- CSV Logging Setup ---
LOG_FILE_PATH = Path(__file__).parent / 'usage_log.csv'
CSV_HEADER = [
    "Timestamp", "Session ID", "Filename", "Filesize (Bytes)", "Request Type", "Model Used",
    "Input Tokens", "Output Tokens", "Grounded Searches Performed",
    "Token Cost USD", "Fixed Cost USD", "Total Estimated Cost USD"
]
csv_lock = threading.Lock()

def log_usage_to_csv(
    timestamp, session_id, filename, filesize, request_type, model_used,
    input_tokens, output_tokens, grounded_searches,
    token_cost, fixed_cost, total_cost
):
    file_exists = LOG_FILE_PATH.is_file()
    with csv_lock:
        try:
            with open(LOG_FILE_PATH, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists or os.path.getsize(LOG_FILE_PATH) == 0:
                    writer.writerow(CSV_HEADER)
                writer.writerow([
                    timestamp.isoformat(),
                    session_id if session_id else "N/A", # Handle optional session_id
                    filename,
                    filesize,
                    request_type,
                    model_used,
                    input_tokens,
                    output_tokens,
                    grounded_searches,
                    f"{token_cost:.8f}",
                    f"{fixed_cost:.8f}",
                    f"{total_cost:.8f}"
                ])
        except IOError as e:
            logger.error(f"Failed to write to CSV log file {LOG_FILE_PATH}: {e}")
        except Exception as e:
             logger.error(f"An unexpected error occurred during CSV logging: {e}", exc_info=True)

# --- Pydantic Models for Request Body (Define before use) ---
class ProductSearchRequest(BaseModel):
    product_name: str
    state_name: Optional[str] = None
    manufacturers: Optional[List[str]] = None
    query: Optional[str] = None

class DownloadRequest(BaseModel):
    urls: List[str]

class PDFLinkExtractionRequest(BaseModel):
    product_name: str
    technical_specifications: dict
    manufacturers: dict
    reference: str = ""
    refresh: bool = False  # Add refresh parameter to bypass cache
    preferred_state: Optional[str] = None  # Add preferred state for manufacturer location preference

class PDFValidationItem(BaseModel):
    pdf_url: str
    product_data: Dict[str, Any]
    filename: Optional[str] = None

class AlreadyValidatedPDFItem(BaseModel):
    pdf_url: str
    product_data: Dict[str, Any]
    filename: Optional[str] = None
    validation_data: Dict[str, Any]  # Existing validation results

class AddValidatedPDFsRequest(BaseModel):
    new_pdfs: List[PDFValidationItem] = []  # PDFs that need validation
    already_validated_pdfs: List[AlreadyValidatedPDFItem] = []  # PDFs already validated in session

class GenerateValidationReportRequest(BaseModel):
    validation_data: Dict[str, Any]
    product_name: str
    original_pdf_url: str

class GenerateSmartValidationReportRequest(BaseModel):
    validation_data: Dict[str, Any]
    product_name: str
    original_pdf_bytes: str  # Base64 encoded PDF bytes

class DownloadIndividualPDFRequest(BaseModel):
    pdf_data: str
    filename: str

# Shared httpx client
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with httpx.AsyncClient() as client:
        app.state.httpx_client = client
        yield
    # Client will be closed automatically upon exiting the context manager

# Initialize FastAPI app with configuration
app = FastAPI(
    root_path="/gemini-prod",
    title="Submittal Factory API",
    description="API for automated submittal validation in construction projects",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(auth_router)
app.include_router(roles_router)
app.include_router(user_roles_router)
app.include_router(users_router)
app.include_router(audit_router)
app.include_router(pds_add_more_router)
app.include_router(subscription_router)

@app.on_event("startup")
async def startup():
    init_db()

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    # Start background task scheduler
    scheduler = start_background_scheduler()
    
    # Ensure proper shutdown
    atexit.register(lambda: scheduler.shutdown())
    
    print("✅ Background scheduler started - checking expired trials every hour")

@app.on_event("shutdown")
async def shutdown():
    close_db_connection()

@app.middleware("http")
async def db_session_middleware(request: Request, call_next):
    '''
    Database session middleware - creates and manages DB sessions per request
    ✅ OPTIMIZED: Better error handling and connection cleanup
    '''
    db = None
    db_generator = None
    
    try:
        db_generator = get_db()
        db = next(db_generator)
        request.state.db = db
        
        # Process the request
        response = await call_next(request)
        return response
        
    except Exception as e:
        if db:
            try:
                db.rollback()
            except Exception:
                pass  # Ignore rollback errors
        logger.error(f"Request error: {e}")
        raise e
        
    finally:
        if db:
            try:
                db.close()
            except Exception:
                pass  # Ignore close errors
        
        if db_generator:
            try:
                next(db_generator)  # Exhaust generator
            except StopIteration:
                pass  # Expected
            except Exception:
                pass  # Ignore cleanup errors

@app.middleware("http")
async def subscription_check_middleware(request: Request, call_next):
    return await check_subscription_middleware(request, call_next)

# Initialize rate limiter for API protection
def get_session_or_ip(request: Request):
    """Get session ID from headers or fall back to IP address for rate limiting."""
    session_id = request.headers.get("x-session-id")
    if session_id:
        return f"session:{session_id}"
    return get_remote_address(request)

limiter = Limiter(key_func=get_session_or_ip)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- Middleware (Keep this early) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


# Add request logging middleware for debugging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    '''
    Request logging middleware
    ✅ OPTIMIZED: Reduced logging to decrease I/O overhead
    '''
    start_time = time.time()
    path = request.url.path
    
    # Only log non-health endpoints to reduce noise
    should_log = not path.endswith('/health') and not path.endswith('/health/db')
    
    if should_log:
        logger.info(f"🔄 {request.method} {path}")
    
    try:
        response = await call_next(request)
        
        if should_log:
            process_time = time.time() - start_time
            logger.info(f"✅ {request.method} {path} - {response.status_code} ({process_time:.2f}s)")
        
        # Add CORS headers
        origin = request.headers.get("origin")
        if origin:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"❌ {request.method} {path} - Error: {str(e)[:100]} ({process_time:.2f}s)")
        
        origin = request.headers.get("origin", "*")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(e)[:200]}"},
            headers={
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Credentials": "true"
            }
        )


# --- API Endpoints (Define FIRST) ---

@app.post("/api/audit-log")
async def record_audit_log(
    body: CustomAuditLogRequest,
    request: Request,
    current_user: User = Depends(get_current_active_user),  # Associates log with user
    x_session_id: Optional[str] = Header(None)
):
    import time
    start_time = time.time()

    # Use your existing logger
    await log_api_success(
        request=request,
        action=body.action,
        entity_type=body.entity_type,
        entity_id=body.entity_id,
        metadata={
            **(body.metadata or {}),
            "session_id": x_session_id,
        },
        cost=body.cost,
        start_time=start_time
    )
    return {"success": True, "message": "Audit log recorded."}

# Health check endpoint for debugging connectivity
@app.get("/api/health")
async def health_check():
    '''
    Health check endpoint
    ✅ ENHANCED: Includes basic pool info
    '''
    from database import get_pool_status
    
    try:
        pool_status = get_pool_status()
    except Exception:
        pool_status = {"error": "Could not get pool status"}
    
    return {
        "status": "healthy",
        "message": "Backend server is running",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pool": pool_status
    }

@app.get("/api/health/db")
async def database_health():
    '''
    Database health check with connection pool status
    ✅ Useful for monitoring connection pool issues
    '''
    from database import check_database_health
    return check_database_health()

@app.get("/api/debug/cors")
async def debug_cors():
    """Debug endpoint to check CORS configuration"""
    return {
        "cors_origins": settings.cors_origins,
        "cors_allow_credentials": settings.cors_allow_credentials,
        "cors_allow_methods": settings.cors_allow_methods,
        "cors_allow_headers": settings.cors_allow_headers
    }

@app.options("/api/extract")
async def extract_options():
    """Handle CORS preflight requests for /api/extract endpoint"""
    return {"message": "CORS preflight successful"}


@app.post("/api/extract")
@limiter.limit("5/minute")
async def extract_data(
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    x_session_id: Optional[str] = Header(None),
    fast_mode: bool = Query(True)
):
    """
    Extract product data from a construction specification PDF.
    
    Supports:
    - Standard documents with PART 2 - PRODUCTS
    - Documents with PART 2 - PRODUCT (singular)
    - Non-standard documents (extracts entire document)
    - Documents with products in PART 1, PART 3, or multiple sections
    
    Args:
        file: PDF file upload
        current_user: Authenticated user
        x_session_id: Optional session ID for tracking
        fast_mode: Use lightning-fast preprocessing (default: True)
    
    Returns:
        JSON with extracted products, metadata, and processing info
        
    Raises:
        SubmittalFactoryException: For all handled errors with standardized codes
    """
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    request_start_time = time.time()
    safe_filename = file.filename or "unknown_file.pdf"
    
    # Initialize tracking variables
    warnings_list = []
    notices_list = []
    section_detection_info = {
        "products_section": "PART_2", 
        "confidence": 1.0, 
        "detection_method": "default"
    }
    use_entire_document = False
    cleaned_text = ""
    file_content = b""
    filesize = 0
    
    # Log request start
    mode_indicator = "⚡ FAST" if fast_mode else "📄 STANDARD"
    logger.info(f"{'='*60}")
    logger.info(f"📥 EXTRACTION START [{mode_indicator}]: {safe_filename}")
    logger.info(f"   Session: {x_session_id or 'N/A'}")
    logger.info(f"{'='*60}")

    try:
        # =========================================================================
        # FILE VALIDATION - Enhanced with standardized errors
        # =========================================================================
        
        # Check file extension
        if not file.filename:
            logger.warning(f"❌ No filename provided")
            raise SubmittalFactoryException(
                ErrorCode.INVALID_FILE_FORMAT,
                "No filename provided in upload",
                {"filename": "unknown"}
            )
        
        if not file.filename.lower().endswith('.pdf'):
            logger.warning(f"❌ Invalid file type: {file.filename}")
            raise SubmittalFactoryException(
                ErrorCode.INVALID_FILE_FORMAT,
                f"Invalid file extension: {file.filename}. Only PDF files are allowed.",
                {"filename": file.filename, "extension": file.filename.split('.')[-1] if '.' in file.filename else "none"}
            )

        # =========================================================================
        # READ FILE CONTENT
        # =========================================================================
        try:
            file_content = await file.read()
            filesize = len(file_content)
            await file.seek(0)
            logger.info(f"📄 File read successfully: {filesize:,} bytes ({filesize/1024:.1f} KB)")
        except Exception as e:
            logger.error(f"❌ Failed to read file: {e}")
            raise SubmittalFactoryException(
                ErrorCode.CORRUPTED_FILE,
                f"Could not read uploaded file: {str(e)}",
                {"filename": safe_filename, "error_type": type(e).__name__}
            )

        # Check for empty file
        if not file_content or filesize == 0:
            logger.error("❌ Empty file uploaded")
            raise SubmittalFactoryException(
                ErrorCode.EMPTY_FILE,
                "Uploaded file is empty (0 bytes)",
                {"filename": safe_filename}
            )

        # Check file size limit (5MB)
        MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
        if filesize > MAX_FILE_SIZE:
            logger.error(f"❌ File too large: {filesize} bytes (max: {MAX_FILE_SIZE})")
            raise SubmittalFactoryException(
                ErrorCode.FILE_TOO_LARGE,
                f"File size ({filesize / 1024 / 1024:.1f}MB) exceeds maximum allowed size (5MB)",
                {"filename": safe_filename, "filesize": filesize, "max_size": MAX_FILE_SIZE}
            )

        # Basic PDF validation (check magic bytes)
        if not file_content.startswith(b'%PDF'):
            logger.error("❌ File does not appear to be a valid PDF")
            raise SubmittalFactoryException(
                ErrorCode.CORRUPTED_FILE,
                "File does not appear to be a valid PDF document (invalid header)",
                {"filename": safe_filename}
            )

        # =========================================================================
        # SECTION DETECTION
        # =========================================================================
        if SECTION_DETECTION_AVAILABLE:
            try:
                logger.info("🔍 Running section detection with Gemini...")
                detection_start = time.time()
                
                # This will RAISE exceptions for critical errors - NO FALLBACK
                detection_result = await run_in_threadpool(
                    detect_products_section,
                    pdf_content=file_content,
                    use_llm=True
                )
                
                detection_time = time.time() - detection_start
                
                section_detection_info = {
                    "products_section": detection_result.get("products_section", "PART_2"),
                    "confidence": detection_result.get("confidence", 1.0),
                    "detection_method": detection_result.get("detection_method", "gemini_full_analysis"),
                    "sections_with_products": detection_result.get("sections_with_products", ["PART_2"]),
                    "products_found": detection_result.get("products_found", []),
                    "reasoning": detection_result.get("reasoning", ""),
                }
                
                use_entire_document = detection_result.get("use_entire_document", False)
                
                logger.info(f"🔍 Section detection complete in {detection_time:.2f}s:")
                logger.info(f"   Section: {section_detection_info['products_section']}")
                logger.info(f"   Sections with products: {section_detection_info['sections_with_products']}")
                logger.info(f"   Confidence: {section_detection_info['confidence']:.2f}")
                logger.info(f"   Method: {section_detection_info['detection_method']}")
                logger.info(f"   Use entire document: {use_entire_document}")
                
                # Log products found by Gemini
                if section_detection_info.get("products_found"):
                    logger.info(f"   Products found by Gemini:")
                    for pf in section_detection_info["products_found"]:
                        logger.info(f"      - {pf.get('section')}: {pf.get('product_types', [])}")
                        logger.info(f"        Manufacturers: {pf.get('manufacturers_mentioned', [])}")
                
                for warning in detection_result.get("warnings", []):
                    notices_list.append(f"ℹ️ {warning}")
                
                if section_detection_info["products_section"] not in ["PART_2", "ENTIRE_DOCUMENT"]:
                    section_display = section_detection_info["products_section"].replace("_", " ")
                    notices_list.append(
                        f"ℹ️ Products detected in {section_display} instead of PART 2. "
                        "Extraction adjusted accordingly."
                    )
            
            # =========================================================================
            # CRITICAL ERROR HANDLING - STOP PROCESSING, RETURN ERROR TO USER
            # =========================================================================
            except ModelNotFoundError as e:
                logger.error(f"❌ CRITICAL: Gemini model not found: {e}")
                raise SubmittalFactoryException(
                    ErrorCode.MODEL_NOT_FOUND,
                    f"Section detection failed: Gemini model not found. Please check SECTION_DETECTION_MODEL environment variable. Error: {str(e)}",
                    {
                        "filename": safe_filename,
                        "error_type": "ModelNotFoundError",
                        "critical": True,
                        "details": str(e)
                    }
                )
            
            except APIKeyNotConfiguredError as e:
                logger.error(f"❌ CRITICAL: Gemini API key not configured: {e}")
                raise SubmittalFactoryException(
                    ErrorCode.API_KEY_INVALID,
                    f"Section detection failed: GOOGLE_API_KEY environment variable not configured. Error: {str(e)}",
                    {
                        "filename": safe_filename,
                        "error_type": "APIKeyNotConfiguredError",
                        "critical": True,
                        "details": str(e)
                    }
                )
            
            except GeminiNotAvailableError as e:
                logger.error(f"❌ CRITICAL: Gemini library not available: {e}")
                raise SubmittalFactoryException(
                    ErrorCode.MODEL_NOT_FOUND,
                    f"Section detection failed: Google Generative AI library not installed. Run: pip install google-generativeai. Error: {str(e)}",
                    {
                        "filename": safe_filename,
                        "error_type": "GeminiNotAvailableError",
                        "critical": True,
                        "details": str(e)
                    }
                )
            
            except GeminiAPIError as e:
                logger.error(f"❌ CRITICAL: Gemini API error: {e}")
                raise SubmittalFactoryException(
                    ErrorCode.MODEL_NOT_FOUND,
                    f"Section detection failed: Gemini API returned an error. Error: {str(e)}",
                    {
                        "filename": safe_filename,
                        "error_type": "GeminiAPIError",
                        "critical": True,
                        "details": str(e)
                    }
                )
            
            except PDFExtractionError as e:
                logger.error(f"❌ CRITICAL: PDF extraction failed during section detection: {e}")
                raise SubmittalFactoryException(
                    ErrorCode.PDF_NOT_READABLE,
                    f"Section detection failed: Could not extract text from PDF for analysis. Error: {str(e)}",
                    {
                        "filename": safe_filename,
                        "error_type": "PDFExtractionError",
                        "critical": True,
                        "details": str(e)
                    }
                )
            
            except InvalidResponseError as e:
                logger.error(f"❌ CRITICAL: Invalid Gemini response: {e}")
                raise SubmittalFactoryException(
                    ErrorCode.LLM_RESPONSE_INVALID,
                    f"Section detection failed: Gemini returned invalid/unparseable response. Error: {str(e)}",
                    {
                        "filename": safe_filename,
                        "error_type": "InvalidResponseError",
                        "critical": True,
                        "details": str(e)
                    }
                )
            
            except SectionDetectionError as e:
                # Catch-all for any other section detection errors
                logger.error(f"❌ CRITICAL: Section detection error: {e}")
                raise SubmittalFactoryException(
                    ErrorCode.EXTRACTION_FAILED,
                    f"Section detection failed: {str(e)}",
                    {
                        "filename": safe_filename,
                        "error_type": type(e).__name__,
                        "critical": True,
                        "details": str(e)
                    }
                )
            
            except Exception as e:
                # Unexpected errors - still stop processing
                logger.error(f"❌ CRITICAL: Unexpected section detection error: {e}")
                logger.error(f"   Error type: {type(e).__name__}")
                raise SubmittalFactoryException(
                    ErrorCode.EXTRACTION_FAILED,
                    f"Section detection failed with unexpected error: {str(e)}",
                    {
                        "filename": safe_filename,
                        "error_type": type(e).__name__,
                        "critical": True,
                        "details": str(e)
                    }
                )
        else:
            # Section detection module not available - this is also a critical error
            logger.error("❌ CRITICAL: Section detection service not available (module not imported)")
            raise SubmittalFactoryException(
                ErrorCode.EXTRACTION_FAILED,
                "Section detection service is not available. Please ensure uploadServices.section_detection_service module is installed.",
                {
                    "filename": safe_filename,
                    "error_type": "ServiceNotAvailable",
                    "critical": True
                }
            )

        # =========================================================================
        # TEXT EXTRACTION - Enhanced error handling with SECTION-AWARE EXTRACTION
        # =========================================================================
        try:
            extraction_start = time.time()
            
            if use_entire_document:
                # -----------------------------------------------------------------
                # ENTIRE DOCUMENT EXTRACTION
                # -----------------------------------------------------------------
                logger.info("📄 Using entire document extraction (non-standard format)")
                
                try:
                    cleaned_text = await _extract_entire_document(file_content, safe_filename)
                except Exception as e:
                    error_str = str(e).lower()
                    if "scanned" in error_str or "image" in error_str or "ocr" in error_str:
                        raise SubmittalFactoryException(
                            ErrorCode.SCANNED_DOCUMENT,
                            f"Document appears to be scanned/image-based: {e}",
                            {"filename": safe_filename}
                        )
                    raise SubmittalFactoryException(
                        ErrorCode.PDF_NOT_READABLE,
                        f"Could not extract text from entire document: {e}",
                        {"filename": safe_filename}
                    )
                
                section_detection_info["products_section"] = "ENTIRE_DOCUMENT"
                notices_list.append(
                    "📄 This document uses a non-standard format. "
                    "Products extracted from entire document."
                )
                
            elif fast_mode:
                # -----------------------------------------------------------------
                # FAST MODE EXTRACTION - ✅ NOW SECTION-AWARE
                # -----------------------------------------------------------------
                logger.info("⚡ Using lightning-fast preprocessing pipeline")
                
                # ✅ FIX: Get the extraction patterns from section detection
                start_pattern, end_pattern = get_section_extraction_patterns(
                    section_detection_info["products_section"]
                )
                
                logger.info(f"⚡ Extracting from detected section: {section_detection_info['products_section']}")
                logger.info(f"   Start pattern: {start_pattern[:50] if start_pattern else 'None'}...")
                logger.info(f"   End pattern: {end_pattern[:50] if end_pattern else 'None'}...")
                
                try:
                    # ✅ FIX: Use the section-aware pipeline with detected patterns
                    preprocessing_result = await run_in_threadpool(
                        lightning_fast_preprocessing_pipeline_with_patterns,
                        file_content,
                        safe_filename,
                        start_pattern=start_pattern,
                        end_pattern=end_pattern
                    )
                except Exception as e:
                    logger.error(f"❌ Fast preprocessing crashed: {e}")
                    raise SubmittalFactoryException(
                        ErrorCode.EXTRACTION_FAILED,
                        f"Fast preprocessing pipeline failed: {e}",
                        {"filename": safe_filename, "mode": "fast"}
                    )
                
                if preprocessing_result["success"]:
                    cleaned_text = preprocessing_result["cleaned_text"]
                    preprocessing_time = preprocessing_result["processing_time"]
                    logger.info(f"⚡ Fast preprocessing completed in {preprocessing_time:.2f}s")
                else:
                    error_msg = preprocessing_result.get("error_message", "Unknown preprocessing error")
                    
                    # Check if the detected section wasn't found
                    if "not found" in error_msg.lower():
                        logger.warning(f"⚠️ Fast preprocessing failed: {error_msg}")
                        logger.info("🔄 Falling back to entire document extraction...")
                        
                        try:
                            cleaned_text = await _extract_entire_document(file_content, safe_filename)
                        except Exception as fallback_err:
                            raise SubmittalFactoryException(
                                ErrorCode.NO_PART2_FOUND,
                                f"Detected section not found and entire document extraction failed: {fallback_err}",
                                {
                                    "filename": safe_filename,
                                    "detected_section": section_detection_info['products_section'],
                                    "original_error": error_msg
                                }
                            )
                        
                        use_entire_document = True
                        section_detection_info["products_section"] = "ENTIRE_DOCUMENT"
                        notices_list.append(
                            "🔄 The detected section could not be extracted. "
                            "Products extracted from entire document."
                        )
                    else:
                        # Check for specific error types
                        error_lower = error_msg.lower()
                        if "scanned" in error_lower or "image" in error_lower:
                            raise SubmittalFactoryException(
                                ErrorCode.SCANNED_DOCUMENT,
                                error_msg,
                                {"filename": safe_filename}
                            )
                        elif "empty" in error_lower or "no text" in error_lower:
                            raise SubmittalFactoryException(
                                ErrorCode.NO_TEXT_CONTENT,
                                error_msg,
                                {"filename": safe_filename}
                            )
                        else:
                            raise SubmittalFactoryException(
                                ErrorCode.EXTRACTION_FAILED,
                                error_msg,
                                {"filename": safe_filename}
                            )
            else:
                # -----------------------------------------------------------------
                # TRADITIONAL MODE EXTRACTION - ✅ ALSO SECTION-AWARE NOW
                # -----------------------------------------------------------------
                logger.info("📄 Using traditional preprocessing pipeline")
                
                # ✅ Get patterns for detected section
                start_pattern, end_pattern = get_section_extraction_patterns(
                    section_detection_info["products_section"]
                )
                
                logger.info(f"📄 Extracting from detected section: {section_detection_info['products_section']}")
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_pdf_path = os.path.join(temp_dir, safe_filename)
                    
                    async with aiofiles.open(temp_pdf_path, "wb") as buffer:
                        await buffer.write(file_content)

                    # ✅ Check if detected section exists (not just PART 2)
                    try:
                        from lightning_fast_preprocessing import lightning_fast_check_part2_exists
                        # This checks for the actual start_pattern
                        section_exists = await run_in_threadpool(
                            lightning_fast_check_part2_exists, 
                            temp_pdf_path
                        )
                    except Exception as e:
                        logger.warning(f"⚠️ Section check failed: {e}")
                        section_exists = False
                    
                    if not section_exists:
                        logger.warning(f"⚠️ Detected section not found in {safe_filename}")
                        logger.info("🔄 Falling back to entire document extraction...")
                        
                        try:
                            cleaned_text = await _extract_entire_document(file_content, safe_filename)
                        except Exception as fallback_err:
                            raise SubmittalFactoryException(
                                ErrorCode.NO_PART2_FOUND,
                                f"Detected section not found and entire document extraction failed: {fallback_err}",
                                {"filename": safe_filename}
                            )
                        
                        use_entire_document = True
                        section_detection_info["products_section"] = "ENTIRE_DOCUMENT"
                        notices_list.append(
                            "📄 Detected section not found in document. "
                            "Products extracted from entire document."
                        )
                    else:
                        try:
                            # ✅ Use detected patterns instead of hardcoded PART_2
                            raw_text = await run_in_threadpool(
                                extract_text_between_patterns, 
                                temp_pdf_path, 
                                start_pattern,
                                end_pattern
                            )
                        except Exception as e:
                            logger.error(f"❌ Pattern extraction failed: {e}")
                            raise SubmittalFactoryException(
                                ErrorCode.EXTRACTION_FAILED,
                                f"Failed to extract detected section: {e}",
                                {"filename": safe_filename}
                            )
                        
                        if not raw_text:
                            logger.warning(f"⚠️ Failed to extract section text, trying entire document...")
                            try:
                                cleaned_text = await _extract_entire_document(file_content, safe_filename)
                            except Exception as fallback_err:
                                raise SubmittalFactoryException(
                                    ErrorCode.NO_TEXT_CONTENT,
                                    f"Section extraction returned empty and fallback failed: {fallback_err}",
                                    {"filename": safe_filename}
                                )
                            use_entire_document = True
                            section_detection_info["products_section"] = "ENTIRE_DOCUMENT"
                        else:
                            try:
                                cleaned_text = await run_in_threadpool(clean_text, raw_text)
                            except Exception as e:
                                logger.error(f"❌ Text cleaning failed: {e}")
                                raise SubmittalFactoryException(
                                    ErrorCode.EXTRACTION_FAILED,
                                    f"Text cleaning failed: {e}",
                                    {"filename": safe_filename}
                                )

            extraction_time = time.time() - extraction_start
            logger.info(f"📝 Text extraction completed in {extraction_time:.2f}s")
            logger.info(f"   Extracted text length: {len(cleaned_text):,} characters")

            # Validate extracted text
            if not cleaned_text or not cleaned_text.strip():
                logger.error("❌ Text extraction resulted in empty content")
                raise SubmittalFactoryException(
                    ErrorCode.NO_TEXT_CONTENT,
                    "Text extraction resulted in empty content. The PDF may be image-based or corrupted.",
                    {"filename": safe_filename, "extraction_method": "entire_document" if use_entire_document else "section"}
                )

            # Check for minimum content
            MIN_TEXT_LENGTH = 50
            if len(cleaned_text.strip()) < MIN_TEXT_LENGTH:
                logger.warning(f"⚠️ Extracted text is very short: {len(cleaned_text)} chars")
                raise SubmittalFactoryException(
                    ErrorCode.NO_TEXT_CONTENT,
                    f"Extracted text is too short ({len(cleaned_text)} characters). The PDF may not contain sufficient product information.",
                    {"filename": safe_filename, "text_length": len(cleaned_text)}
                )

        except SubmittalFactoryException:
            raise  # Re-raise our custom exceptions
        except Exception as e:
            logger.error(f"❌ Unexpected error during text extraction: {e}", exc_info=True)
            error_code = classify_extraction_error(e)
            raise SubmittalFactoryException(
                error_code,
                f"Text extraction failed: {e}",
                {"filename": safe_filename, "error_type": type(e).__name__}
            )

        # =====================================================================
        # JSON STRUCTURE GENERATION (Optional - non-critical)
        # =====================================================================
        try:
            json_structure = await run_in_threadpool(parse_text_to_json_structure, cleaned_text)
            if not json_structure:
                logger.debug("ℹ️ JSON structure generation returned empty (non-critical)")
        except Exception as e:
            logger.warning(f"⚠️ JSON structure generation failed (non-critical): {e}")
            json_structure = None

        # =====================================================================
        # METADATA EXTRACTION (Skip in fast mode)
        # =====================================================================
        metadata_dict = {}
        meta_prompt = 0
        meta_completion = 0
        meta_model = METADATA_MODEL_NAME

        if not fast_mode:
            logger.info("📋 Generating metadata...")
            metadata_start = time.time()
            
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_pdf_path = os.path.join(temp_dir, safe_filename)
                    async with aiofiles.open(temp_pdf_path, "wb") as buffer:
                        await buffer.write(file_content)
                    
                    metadata_result = await run_in_threadpool(
                        generate_metadata_and_get_tokens, 
                        temp_pdf_path
                    )
                    metadata_dict = metadata_result.get("data", {})
                    meta_prompt = metadata_result.get("prompt_tokens", 0)
                    meta_completion = metadata_result.get("completion_tokens", 0)
                    meta_model = metadata_result.get("model", METADATA_MODEL_NAME)
                    
                metadata_time = time.time() - metadata_start
                logger.info(f"📋 Metadata extraction completed in {metadata_time:.2f}s")
                
                if not metadata_dict:
                    logger.warning("⚠️ Metadata generation returned empty")
            except Exception as e:
                # Metadata extraction failure is non-critical
                logger.warning(f"⚠️ Metadata extraction failed (non-critical): {e}")
                notices_list.append("ℹ️ Metadata extraction was skipped due to processing issues.")
        else:
            logger.info("⚡ Skipping metadata extraction (fast mode)")

        # =====================================================================
        # TEXT MINIFICATION (For long documents)
        # =====================================================================
        text_for_extraction = cleaned_text
        minification_used = False
        
        MINIFICATION_THRESHOLD = 10000
        
        if len(cleaned_text) > MINIFICATION_THRESHOLD:
            logger.info(f"✂️ Text is long ({len(cleaned_text):,} chars), attempting minification...")
            
            try:
                from part2_minify import minify_part2_with_llm
                from pathlib import Path
                
                minify_start = time.time()
                
                minified_text, used_fallback = await run_in_threadpool(
                    minify_part2_with_llm,
                    cleaned_text,
                    Path(safe_filename).stem
                )
                
                minify_time = time.time() - minify_start
                reduction_pct = (1 - len(minified_text) / len(cleaned_text)) * 100
                
                logger.info(f"✂️ Minification completed in {minify_time:.2f}s:")
                logger.info(f"   Original: {len(cleaned_text):,} chars")
                logger.info(f"   Minified: {len(minified_text):,} chars")
                logger.info(f"   Reduction: {reduction_pct:.1f}%")
                logger.info(f"   Method: {'fallback' if used_fallback else 'LLM'}")
                
                text_for_extraction = minified_text
                minification_used = True
                
                if used_fallback:
                    notices_list.append(
                        "ℹ️ Text minification used fallback method due to LLM limitations."
                    )
                    
            except ImportError:
                logger.warning("⚠️ Minification module not available, using original text")
            except Exception as minify_err:
                logger.warning(f"⚠️ Minification failed: {minify_err}")
                logger.warning("   Continuing with original text")
                # Don't raise - continue with original text
        else:
            logger.info(f"📝 Text is short ({len(cleaned_text):,} chars), skipping minification")

        # =====================================================================
        # PRODUCT DATA EXTRACTION (LLM) - Critical section with enhanced error handling
        # =====================================================================
        logger.info("🤖 Extracting product data with LLM...")
        llm_start = time.time()
        
        try:
            product_extraction_result = await run_in_threadpool(
                extract_product_data, 
                text_for_extraction
            )
        except SubmittalFactoryException:
            # Already a standardized app error -> just bubble up
            raise
        except Exception as e:
            # Truly unexpected LLM or infrastructure error
            logger.error(f"❌ LLM extraction failed: {e}", exc_info=True)
            error_code = classify_llm_error(e)
            
            # Log the classification for debugging
            logger.error(
                f"   Classified as: {error_code.value} - "
                f"{ERROR_MESSAGES.get(error_code, {}).get('user_message', 'Unknown')}"
            )
            
            # Wrap in our standardized exception for the API response
            raise SubmittalFactoryException(
                error_code,
                str(e),
                {
                    "filename": safe_filename,
                    "original_error_type": type(e).__name__,
                    "text_length": len(text_for_extraction)
                }
            )

        # 👉 NEW: handle standardized error returned from extract_product_data()
        if product_extraction_result.get("error"):
            err = product_extraction_result["error"]

            # error_code is a string like "3001"
            err_code_str = err.get(
                "error_code",
                ErrorCode.LLM_RESPONSE_INVALID.value
            )

            try:
                # Convert "3001" → ErrorCode.MODEL_NOT_FOUND, etc.
                err_code = ErrorCode(err_code_str)
            except ValueError:
                err_code = ErrorCode.LLM_RESPONSE_INVALID

            # Raise a SubmittalFactoryException with original LLM error details
            raise SubmittalFactoryException(
                err_code,
                err.get("error_message", "LLM extraction failed."),
                {
                    "filename": safe_filename,
                    "text_length": len(text_for_extraction),
                    "llm_details": err.get("details", {}),
                }
            )
        
        # Only reach here if:
        #  - no Python exception AND
        #  - extract_product_data() did NOT return an "error" object
        llm_time = time.time() - llm_start
        
        # Extract results
        json_output_str_or_dict = product_extraction_result.get('data')
        prod_prompt = product_extraction_result.get('prompt_tokens', 0)
        prod_completion = product_extraction_result.get('completion_tokens', 0)
        prod_model = product_extraction_result.get('model', LLM_MODEL_NAME)
        retry_attempts = product_extraction_result.get('retry_attempts', 0)
        truncated = product_extraction_result.get('truncated', False)
        extraction_notices = product_extraction_result.get('notices', [])
        display_model_name = prod_model
        
        logger.info(f"🤖 LLM extraction completed in {llm_time:.2f}s:")
        logger.info(f"   Model: {display_model_name}")
        logger.info(f"   Input tokens: {prod_prompt:,}")
        logger.info(f"   Output tokens: {prod_completion:,}")
        logger.info(f"   Retry attempts: {retry_attempts}")
        logger.info(f"   Truncated: {truncated}")
        
        # Validate LLM response
        if json_output_str_or_dict is None:
            logger.error("❌ LLM returned None for product data")
            raise SubmittalFactoryException(
                ErrorCode.LLM_RESPONSE_INVALID,
                "LLM returned empty response for product extraction",
                {"filename": safe_filename, "model": display_model_name}
            )
        
        if retry_attempts > 1:
            logger.info(f"ℹ️ Product extraction required {retry_attempts} attempts")
            notices_list.append(
                f"ℹ️ Extraction required {retry_attempts} attempts due to response complexity."
            )
        
        if truncated:
            warnings_list.append({"type": "truncated_response", "severity": "warning"})
            notices_list.append(
                "⚠️ Response was truncated due to length. Some products may be missing."
            )
        
        notices_list.extend(extraction_notices)

        # =====================================================================
        # COST CALCULATION
        # =====================================================================
        meta_token_cost = get_token_cost(meta_model, meta_prompt, meta_completion) if not fast_mode else 0.0
        prod_token_cost = get_token_cost(prod_model, prod_prompt, prod_completion)
        
        combined_input_tokens = meta_prompt + prod_prompt
        combined_output_tokens = meta_completion + prod_completion
        combined_token_cost = meta_token_cost + prod_token_cost
        combined_total_cost = combined_token_cost
        
        logger.info(f"💰 Cost calculation:")
        logger.info(f"   Total tokens: {combined_input_tokens + combined_output_tokens:,}")
        logger.info(f"   Total cost: ${combined_total_cost:.6f}")

        # =====================================================================
        # CSV LOGGING
        # =====================================================================
        current_time = datetime.datetime.now(datetime.timezone.utc)
        request_type = "fast_llm_extraction" if fast_mode else "standard_llm_extraction"
        
        try:
            await run_in_threadpool(
                log_usage_to_csv,
                timestamp=current_time,
                session_id=x_session_id,
                filename=safe_filename,
                filesize=filesize,
                request_type=request_type,
                model_used=display_model_name,
                input_tokens=combined_input_tokens,
                output_tokens=combined_output_tokens,
                grounded_searches=0,
                token_cost=combined_token_cost,
                fixed_cost=0.0,
                total_cost=combined_total_cost
            )
        except Exception as csv_err:
            logger.error(f"⚠️ CSV logging failed: {csv_err}")
            # Non-critical, continue

        # =====================================================================
        # PROCESS PRODUCTS LIST
        # =====================================================================
        logger.info("📦 Processing product data...")
        products_list = []
        
        if isinstance(json_output_str_or_dict, dict) and "products" in json_output_str_or_dict:
            raw_products = json_output_str_or_dict["products"]
        elif isinstance(json_output_str_or_dict, list):
            raw_products = json_output_str_or_dict
        else:
            logger.warning(f"⚠️ Unexpected product data format: {type(json_output_str_or_dict)}")
            raw_products = []

        raw_product_count = len(raw_products) if isinstance(raw_products, list) else 0
        logger.info(f"   Raw products from LLM: {raw_product_count}")

        # Check if no products were extracted
        if raw_product_count == 0:
            logger.warning("⚠️ No products extracted from LLM response")
            raise SubmittalFactoryException(
                ErrorCode.LLM_RESPONSE_INVALID,
                "No products could be extracted from the document. Ensure product data is clearly listed in the specification.",
                {
                    "filename": safe_filename,
                    "text_length": len(text_for_extraction),
                    "response_type": type(json_output_str_or_dict).__name__
                }
            )

        if fast_mode and raw_products:
            try:
                products_list = await run_in_threadpool(
                    lightning_fast_prepare_product_data, 
                    raw_products
                )
            except Exception as e:
                logger.warning(f"⚠️ Fast product processing failed: {e}, using traditional method")
                products_list = _process_raw_products(raw_products)
        else:
            products_list = _process_raw_products(raw_products)

        logger.info(f"📦 Final processed products: {len(products_list)}")
        
        if len(products_list) != raw_product_count:
            logger.warning(
                f"⚠️ Product count mismatch: Raw={raw_product_count}, "
                f"Processed={len(products_list)}"
            )

        # =====================================================================
        # PRODUCT COUNT WARNINGS
        # =====================================================================
        product_count = len(products_list)
        
        if product_count >= PRODUCT_COUNT_CRITICAL_THRESHOLD:
            logger.warning(f"⚠️ Large product count detected: {product_count}")
            notices_list.append(
                f"⚠️ Large Document Warning: {product_count} products extracted. "
                "Some details may be incomplete due to document size."
            )
            warnings_list.append({
                "type": "large_product_count", 
                "severity": "critical",
                "count": product_count
            })
        elif product_count >= PRODUCT_COUNT_WARNING_THRESHOLD:
            logger.info(f"ℹ️ High product count: {product_count}")
            notices_list.append(
                f"📋 Note: {product_count} products extracted. Please review for completeness."
            )
            warnings_list.append({
                "type": "large_product_count", 
                "severity": "warning",
                "count": product_count
            })

        # Check for >100 products (SOW limit)
        if product_count > 100:
            notices_list.append(
                "⚠️ The uploaded file contains more than 100 products. "
                "Accuracy may be affected for very large specifications."
            )

        # =====================================================================
        # CALCULATE TOTAL PROCESSING TIME
        # =====================================================================
        total_processing_time = time.time() - request_start_time
        process_time_ms = int(total_processing_time * 1000)
        
        logger.info(f"{'='*60}")
        logger.info(f"✅ EXTRACTION COMPLETE: {safe_filename}")
        logger.info(f"   Products extracted: {len(products_list)}")
        logger.info(f"   Total time: {total_processing_time:.2f}s")
        logger.info(f"   Total cost: ${combined_total_cost:.6f}")
        logger.info(f"   Warnings: {len(warnings_list)}")
        logger.info(f"   Notices: {len(notices_list)}")
        logger.info(f"{'='*60}")

        # =====================================================================
        # AUDIT LOGGING
        # =====================================================================
        await log_api_success(
            request=request,
            action="Extract",
            entity_type="PDF",
            entity_id=safe_filename,
            metadata={
                "filename": safe_filename,
                "filesize": filesize,
                "fast_mode": fast_mode,
                "products_count": len(products_list),
                "model_used": display_model_name,
                "input_tokens": combined_input_tokens,
                "output_tokens": combined_output_tokens,
                "use_entire_document": use_entire_document,
                "minification_used": minification_used,
                "section_detected": section_detection_info["products_section"]
            },
            cost=combined_total_cost,
            start_time=request_start_time
        )

        # =====================================================================
        # GET SUBSCRIPTION STATUS
        # =====================================================================
        try:
            from services.subscription_service import SubscriptionService
            subscription_status = SubscriptionService.get_subscription_status(
                request.state.db, 
                current_user.user_id
            )
        except Exception as sub_err:
            logger.warning(f"⚠️ Could not get subscription status: {sub_err}")
            subscription_status = {
                'api_calls_used': 0,
                'api_calls_remaining': 0,
                'display_message': 'Subscription status unavailable'
            }

        # =====================================================================
        # RETURN SUCCESS RESPONSE
        # =====================================================================
        return {
            # Core data
            "success": True,
            "products": products_list,
            "part2_text": cleaned_text,
            
            # Model info
            "model_name": display_model_name,
            "input_tokens": combined_input_tokens,
            "output_tokens": combined_output_tokens,
            "total_cost": combined_total_cost,
            
            # Processing info
            "processing_time": round(total_processing_time, 2),
            "fast_mode": fast_mode,
            "truncated": truncated,
            
            # Section detection
            "section_detection": section_detection_info,
            
            # Warnings and notices
            "warnings": warnings_list,
            "notices": notices_list,
            "has_warnings": len(warnings_list) > 0,
            
            # Subscription info
            "subscription": {
                "uploads_used": subscription_status.get('api_calls_used', 0),
                "uploads_remaining": subscription_status.get('api_calls_remaining', 0),
                "message": subscription_status.get('display_message', '')
            }
        }

    # =========================================================================
    # ERROR HANDLING - Enhanced with standardized errors
    # =========================================================================
    except SubmittalFactoryException as sf_exc:
        # Our custom exception - convert to proper response
        logger.error(f"❌ SubmittalFactoryException: {sf_exc.error_code.value} - {sf_exc.technical_message}")
        await log_api_failure(
            request=request,
            action="Extract",
            entity_type="PDF",
            entity_id=safe_filename,
            error=f"{sf_exc.error_code.value}: {sf_exc.technical_message}",
            start_time=request_start_time
        )
        # Return standardized error response
        return sf_exc.to_response(include_technical=getattr(settings, 'debug', False))
    
    except HTTPException as http_exc:
        # Legacy HTTPException - convert to standardized format
        logger.error(f"❌ HTTP Error: {http_exc.status_code} - {http_exc.detail}")
        await log_api_failure(
            request=request,
            action="Extract",
            entity_type="PDF",
            entity_id=safe_filename,
            error=http_exc,
            start_time=request_start_time
        )
        
        # Convert HTTPException to standardized response
        error_response = create_error_response(
            ErrorCode.INTERNAL_ERROR,
            http_exc.detail,
            {"http_status": http_exc.status_code},
            include_technical=getattr(settings, 'debug', False)
        )
        return JSONResponse(
            status_code=http_exc.status_code,
            content=error_response
        )
        
    except FileNotFoundError as fnf_err:
        logger.error(f"❌ File not found error: {fnf_err}", exc_info=True)
        await log_api_failure(
            request=request,
            action="Extract",
            entity_type="PDF",
            entity_id=safe_filename,
            error=fnf_err,
            start_time=request_start_time
        )
        
        error_response = create_error_response(
            ErrorCode.CORRUPTED_FILE,
            f"Temporary file processing error: {str(fnf_err)}",
            {"filename": safe_filename},
            include_technical=getattr(settings, 'debug', False)
        )
        return JSONResponse(status_code=500, content=error_response)
        
    except Exception as e:
        logger.error(f"❌ Unexpected error during extraction: {e}", exc_info=True)
        try:
            await log_api_failure(
                request=request,
                action="Extract",
                entity_type="PDF",
                entity_id=safe_filename,
                error=str(e),
                start_time=request_start_time
            )
        except Exception as log_error:
            logger.error(f"Failed to log error (ignoring): {log_error}")
        
        # Try to classify the error
        error_str = str(e).lower()
        
        if any(x in error_str for x in ["gemini", "model", "api", "llm"]):
            error_code = classify_llm_error(e)
        elif any(x in error_str for x in ["connection", "timeout", "network"]):
            error_code = classify_network_error(e)
        else:
            error_code = ErrorCode.INTERNAL_ERROR
        
        error_response = create_error_response(
            error_code,
            str(e),
            {"filename": safe_filename, "error_type": type(e).__name__},
            include_technical=getattr(settings, 'debug', False)
        )
        return JSONResponse(status_code=500, content=error_response)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def _extract_entire_document(file_content: bytes, filename: str) -> str:
    """
    Extract all text from a PDF document (for non-standard formats).
    
    Args:
        file_content: PDF file content as bytes
        filename: Original filename for logging
        
    Returns:
        Cleaned text from the entire document
    """
    logger.info(f"📄 Extracting entire document: {filename}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_pdf_path = os.path.join(temp_dir, filename)
        
        async with aiofiles.open(temp_pdf_path, "wb") as buffer:
            await buffer.write(file_content)
        
        raw_text = ""
        
        # Try PyMuPDF first (faster)
        try:
            import fitz
            doc = fitz.open(temp_pdf_path)
            for page_num, page in enumerate(doc):
                page_text = page.get_text("text")
                raw_text += page_text + "\n"
            doc.close()
            logger.info(f"   Extracted {len(doc)} pages using PyMuPDF")
        except Exception as fitz_err:
            logger.warning(f"⚠️ PyMuPDF extraction failed: {fitz_err}")
            
            # Fallback to pypdf
            try:
                reader = PdfReader(temp_pdf_path)
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    raw_text += page_text + "\n"
                logger.info(f"   Extracted {len(reader.pages)} pages using pypdf")
            except Exception as pypdf_err:
                logger.error(f"❌ pypdf extraction also failed: {pypdf_err}")
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract text from PDF using any available method."
                )
        
        if not raw_text.strip():
            raise HTTPException(
                status_code=400,
                detail="The PDF appears to be empty or contains only images/scanned content."
            )
        
        # Clean the extracted text
        cleaned = await run_in_threadpool(clean_text, raw_text)
        
        logger.info(f"   Extracted text length: {len(cleaned):,} characters")
        
        return cleaned


def _process_raw_products(raw_products: list) -> list:
    """
    Process raw product data from LLM into standardized format.
    
    Args:
        raw_products: List of raw product dictionaries from LLM
        
    Returns:
        List of processed product dictionaries
    """
    products_list = []
    
    for product_data in raw_products:
        if not isinstance(product_data, dict):
            logger.warning(f"⚠️ Skipping non-dict product item: {type(product_data)}")
            continue
        
        # Process technical specifications
        tech_specs = []
        if "technical_specifications" in product_data:
            specs = product_data.get("technical_specifications", [])
            if isinstance(specs, list):
                for spec_obj in specs:
                    if isinstance(spec_obj, dict):
                        for key, value in spec_obj.items():
                            tech_specs.append(f"{key}: {value}")
                    elif isinstance(spec_obj, str):
                        tech_specs.append(spec_obj)
            elif isinstance(specs, dict):
                for key, value in specs.items():
                    tech_specs.append(f"{key}: {value}")
        
        # Process manufacturers
        manufacturers = []
        if "manufacturers" in product_data:
            mfrs = product_data.get("manufacturers", {})
            if isinstance(mfrs, dict):
                if "base" in mfrs and isinstance(mfrs["base"], list):
                    manufacturers.extend(mfrs["base"])
                if "optional" in mfrs and isinstance(mfrs["optional"], list):
                    manufacturers.extend(mfrs["optional"])
                # Handle other nested lists
                for key, value in mfrs.items():
                    if key not in ["base", "optional"] and isinstance(value, list):
                        manufacturers.extend(value)
            elif isinstance(mfrs, list):
                manufacturers = mfrs
        
        # Build processed product
        processed_product = {
            "product_name": product_data.get("product_name", ""),
            "technical_specifications": tech_specs,
            "manufacturers": manufacturers,
            "reference": product_data.get("reference", "")
        }
        
        products_list.append(processed_product)
        logger.debug(f"   Processed: {processed_product.get('product_name', 'Unknown')[:50]}")
    
    return products_list

@app.get("/api/preprocessing-info")
async def get_preprocessing_info():
    """Get information about the lightning-fast preprocessing implementation"""
    try:
        stats = await run_in_threadpool(get_preprocessing_stats)
        return {
            "status": "active",
            "implementation": stats["implementation"],
            "features": stats["features"],
            "optimizations": stats["optimizations"],
            "performance_targets": stats["performance_targets"],
            "fast_mode_enabled": True,
            "description": "Lightning-fast preprocessing is integrated and available via fast_mode parameter in /api/extract"
        }
    except Exception as e:
        logger.error(f"Error getting preprocessing info: {e}")
        return {
            "status": "error",
            "error": str(e),
            "fast_mode_enabled": False
        }

# in use , but where not clear
@app.post("/api/search-submittals")
@limiter.limit("20/minute")  # Allow 20 searches per minute per IP
async def search_submittals_endpoint(request: Request, request_data: ProductSearchRequest):
    """Endpoint to search for submittal PDFs using SerpApi."""
    try:
        logger.info(f"Received search request for: {request_data.product_name}, State: {request_data.state_name}")
        results, query_used = await run_in_threadpool(
            search_pdfs_serpapi,
            product_name=request_data.product_name,
            state_name=request_data.state_name,
            manufacturers=request_data.manufacturers,
            num_results=15,
            custom_query=request_data.query
        )
        logger.info(f"SerpApi returned {len(results)} potential results using query: {query_used}")
        return {"results": results, "query_used": query_used}
    except Exception as e:
        logger.error(f"Error during submittal search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during submittal search: {str(e)}")

@app.get("/api/proxy-pdf")
async def proxy_pdf(url: str = Query(...), client: httpx.AsyncClient = Depends(lambda: app.state.httpx_client)):
    logger.info(f"Proxying PDF request for original URL: {url}")
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            logger.warning(f"Invalid URL provided to proxy: {url}")
            raise HTTPException(status_code=400, detail="Invalid URL provided.")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36",
            "Accept": "application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        # Set improved timeout values for large PDFs
        timeout_config = httpx.Timeout(
            connect=30.0,  # 30 seconds to establish connection
            read=180.0,    # 3 minutes to read response
            write=30.0,    # 30 seconds to write
            pool=240.0     # 4 minutes total timeout
        )
        
        response = await client.get(url, headers=headers, follow_redirects=True, timeout=timeout_config)
        
        effective_url = str(response.url)
        if effective_url != url:
            logger.debug(f"Request followed redirect(s) to: {effective_url}")

        response.raise_for_status() 
        
        content_type = response.headers.get("content-type", "").lower()
        logger.info(f"Response from {effective_url}: Status {response.status_code}, Content-Type: {content_type}")

        pdf_content = response.content
        if not pdf_content:
            logger.error(f"No content received from {effective_url} despite a successful status code.")
            raise HTTPException(status_code=502, detail=f"No content received from the target server: {effective_url}")

        # Check for PDF content-type
        if not content_type.startswith("application/pdf"):
            logger.error(f"URL did not return a PDF. Content-Type: {content_type}. Returning error.")
            raise HTTPException(
                status_code=415,
                detail=f"The URL did not return a PDF file. Content-Type: {content_type}."
            )

        return StreamingResponse(io.BytesIO(pdf_content), media_type="application/pdf")

    except httpx.HTTPStatusError as e:
        final_url_attempted = str(e.request.url)
        logger.error(f"HTTP error {e.response.status_code} when fetching from {final_url_attempted} (original URL: {url}). Response: {e.response.text[:200]}...")
        raise HTTPException(status_code=e.response.status_code, detail=f"Error fetching PDF from origin ({final_url_attempted}): Status {e.response.status_code}")
    
    except httpx.TimeoutException as e:
        final_url_attempted = str(e.request.url)
        logger.error(f"Timeout when fetching PDF from {final_url_attempted} (original URL: {url}): {e}")
        raise HTTPException(status_code=504, detail=f"Gateway timeout when fetching PDF from {final_url_attempted}.")
            
    except httpx.RequestError as e:
        final_url_attempted = str(e.request.url) if e.request else url
        logger.error(f"Request error fetching PDF from {final_url_attempted}: {e}")
        raise HTTPException(status_code=502, detail=f"Bad gateway or network error when fetching PDF: {str(e)}")

    except HTTPException: 
        raise
    except Exception as e:
        logger.error(f"Unexpected critical error in proxy_pdf for URL {url}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected internal error occurred while trying to proxy the PDF.")

@app.post("/api/download-pdfs")
async def download_pdfs(request: DownloadRequest, client: httpx.AsyncClient = Depends(lambda: app.state.httpx_client)):
    """Downloads multiple PDFs and returns them as a ZIP file."""
    try:
        # Only log the request and errors
        logger.info(f"Received request to download PDFs: {request.urls}")
        download_tasks = []

        async def download_single_pdf(pdf_url, temp_dir_path, http_client):
            try:
                parsed_url = urlparse(pdf_url)
                if not parsed_url.scheme or not parsed_url.netloc:
                    logger.warning(f"Skipping invalid URL: {pdf_url}")
                    return None

                response = await http_client.get(pdf_url)
                response.raise_for_status()

                filename = os.path.basename(parsed_url.path) or f"{Path(tempfile.mktemp()).name}.pdf"
                temp_file_path = os.path.join(temp_dir_path, filename)
                
                async with aiofiles.open(temp_file_path, "wb") as temp_file:
                    await temp_file.write(response.content)
                
                # Only log on error
                return temp_file_path, filename
            except httpx.RequestError as e:
                logger.error(f"Error downloading PDF from URL {pdf_url}: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error while processing URL {pdf_url}: {e}", exc_info=True)
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            for url in request.urls:
                download_tasks.append(download_single_pdf(url, temp_dir, client))
            
            downloaded_files = await asyncio.gather(*download_tasks)

            zip_path = os.path.join(temp_dir, "downloaded_pdfs.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for result in downloaded_files:
                    if result:
                        temp_file_path, arcname = result
                        zipf.write(temp_file_path, arcname=arcname)
            
            if not any(downloaded_files):
                 logger.warning("No files were successfully downloaded to include in the ZIP.")
                 raise HTTPException(status_code=400, detail="No files could be downloaded.")

            # Read the zip file and return as streaming response
            with open(zip_path, 'rb') as zip_file:
                zip_content = zip_file.read()
            
            return StreamingResponse(
                io.BytesIO(zip_content), 
                media_type="application/zip",
                headers={"Content-Disposition": "attachment; filename=downloaded_pdfs.zip"}
            )
    except Exception as e:
        logger.error(f"Error during PDF download and ZIP creation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during PDF download: {str(e)}")

@app.post("/api/validate-specs")
@limiter.limit("10/minute")  # Allow 10 validations per minute per IP
async def validate_specifications(request: Request, 
                                pdf_url: str = Form(...), 
                                product_data_json: str = Form(...), 
                                client: httpx.AsyncClient = Depends(lambda: app.state.httpx_client),
                                current_user: User = Depends(get_current_active_user),
                                x_session_id: Optional[str] = Header(None)):
    logger.info(f"Received request for /api/validate-specs with URL: {pdf_url}, SessionID: {x_session_id}")
    product_to_validate = None
    temp_pdf_path_for_logging = pdf_url
    filesize_for_logging = 0
    start_time = time.time()

    try: # Outer try for the entire endpoint function
        # Inner try for operations within the temp directory and specific error handling
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    url_hash = hashlib.md5(pdf_url.encode()).hexdigest()
                    safe_filename = f"{url_hash}.pdf"
                except Exception:
                    safe_filename = f"{tempfile.mktemp(suffix='.pdf', dir='', prefix='')}.pdf"
                temp_pdf_path = os.path.join(temp_dir, safe_filename)
                temp_pdf_path_for_logging = safe_filename

                # Download the PDF to a temporary file with retry logic
                max_retries = 3
                retry_delay = 1
                pdf_content_bytes = None  # Cache the PDF content
                download_successful = False
                
                for attempt in range(max_retries):
                    try:
                        # Set more generous timeout values for large PDFs and complex validations
                        timeout_config = httpx.Timeout(
                            connect=60.0,  # 1 minute to establish connection
                            read=300.0,    # 5 minutes to read each chunk  
                            write=60.0,    # 1 minute to write
                            pool=360.0     # 6 minutes total timeout (longer than frontend's 5 minutes)
                        )
                        
                        # Download the PDF to a temporary file with increased timeout and retry logic
                        async with client.stream("GET", pdf_url, headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True, timeout=timeout_config) as response:
                            response.raise_for_status()
                            
                            # Use smaller chunks and avoid storing all chunks in memory
                            total_size = 0
                            async with aiofiles.open(temp_pdf_path, "wb") as buffer:
                                async for chunk in response.aiter_bytes(chunk_size=4096):  # Smaller chunks for better memory efficiency
                                    await buffer.write(chunk)
                                    total_size += len(chunk)
                                    
                                    # Prevent extremely large downloads (> 100MB)
                                    if total_size > 100 * 1024 * 1024:
                                        raise HTTPException(status_code=413, detail="PDF file too large (>100MB)")
                            
                            # Read the file back only when needed for validation
                            async with aiofiles.open(temp_pdf_path, "rb") as f:
                                pdf_content_bytes = await f.read()
                            download_successful = True
                        # If successful, break out of retry loop
                        break
                    except (httpx.RequestError, httpx.TimeoutException, httpx.ReadTimeout) as e:
                        if attempt == max_retries - 1:  # Last attempt
                            logger.error(f"Failed to download PDF after {max_retries} attempts: {e}")
                            # Don't raise here, handle gracefully below
                        else:
                            logger.warning(f"PDF download attempt {attempt + 1} failed, retrying in {retry_delay}s: {e}")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                    except Exception as e:
                        logger.error(f"Unexpected error downloading PDF on attempt {attempt + 1}: {e}")
                        if attempt == max_retries - 1:
                            break
                
                if not download_successful or not pdf_content_bytes:
                    logger.error(f"Unable to download PDF for validation: {pdf_url}")
                    return JSONResponse(
                        status_code=502,
                        content={
                            "success": False,
                            "message": "Failed to download PDF for validation - Looks like this PDF is protected and can't be accessed through our system. No worries — just download it to your device and upload it directly.",
                            "validation_score": 0,
                            "product_name": "Unknown",
                            "specifications_count": None,
                            "summary": None,
                            "unmatched_specifications": None
                        }
                    )
                
                filesize_for_logging = len(pdf_content_bytes) if pdf_content_bytes else 0
                if filesize_for_logging == 0:
                    logger.warning(f"Downloaded PDF is empty for URL: {pdf_url}")
                    return JSONResponse(
                        status_code=502,
                        content={
                            "success": False,
                            "message": "Downloaded PDF was empty",
                            "validation_score": 0,
                            "product_name": "Unknown",
                            "specifications_count": None,
                            "summary": None,
                            "unmatched_specifications": None
                        }
                    )

                try:
                    product_to_validate = json.loads(product_data_json)
                    if not isinstance(product_to_validate, dict) or 'name' not in product_to_validate:
                        raise HTTPException(status_code=422, detail="Invalid product data structure.")
                except json.JSONDecodeError as e_json:
                    raise HTTPException(status_code=422, detail=f"Invalid product data JSON: {e_json}")

                try:
                    from validate_with_llm import validate_pdf_with_llm
                except ImportError:
                    from .validate_with_llm import validate_pdf_with_llm
                
                validation_op_successful, validation_data = await run_in_threadpool(
                    validate_pdf_with_llm,
                    pdf_path=temp_pdf_path,
                    product_data=product_to_validate
                )

                val_prompt_tokens = validation_data.get("prompt_tokens", 0)
                val_completion_tokens = validation_data.get("completion_tokens", 0)
                val_model_used = validation_data.get("model_used", "gemini-2.0-flash")
                
                # Ensure val_model_used is a string, not a dict
                if isinstance(val_model_used, dict):
                    logger.warning(f"val_model_used is a dict: {val_model_used}, using default model")
                    val_model_used = "gemini-2.0-flash"

                val_token_cost = get_token_cost(val_model_used, val_prompt_tokens, val_completion_tokens)
                val_fixed_cost = 0.0
                val_total_cost = val_token_cost + val_fixed_cost
                
                current_time = datetime.datetime.now(datetime.timezone.utc)
                await run_in_threadpool(
                    log_usage_to_csv,
                    timestamp=current_time,
                    session_id=x_session_id,
                    filename=temp_pdf_path_for_logging,
                    filesize=filesize_for_logging,
                    request_type="standard_llm_validation",
                    model_used=val_model_used,
                    input_tokens=val_prompt_tokens,
                    output_tokens=val_completion_tokens,
                    grounded_searches=0,
                    token_cost=val_token_cost,
                    fixed_cost=val_fixed_cost,
                    total_cost=val_total_cost
                )
                logger.info(f"Validation - File: {temp_pdf_path_for_logging}, SessionID: {x_session_id}, Model: {val_model_used}, Input: {val_prompt_tokens}, Output: {val_completion_tokens}, TokenCost: ${val_token_cost:.6f}, TotalCost: ${val_total_cost:.6f}")

                # ============================================================================
                # ✅ CRITICAL FIX: Response Construction
                # ============================================================================
                # Start with the full result from the LLM validation function
                response_payload = validation_data.copy()

                # ✅ FIX: success should indicate if the API operation succeeded, not validation score
                # Low validation scores (0%, 20%, etc.) are valid results, not errors
                response_payload["success"] = validation_op_successful
                
                # Ensure validation_score is present
                response_payload["validation_score"] = validation_data.get("validation_score", 0)
                
                # Construct a meaningful message
                prod_name_for_message = product_to_validate.get('name', 'Unknown Product')
                final_score_for_message = validation_data.get("validation_score", 0)
                unmatched_specs_count = len(validation_data.get("unmatched_specifications", []))
                specs_match_str = validation_data.get("specifications_match", "0/0")
                matched_specs_count = 0
                total_specs_count = 0
                
                if '/' in specs_match_str:
                    try:
                        parts = specs_match_str.split('/')
                        matched_specs_count = int(parts[0])
                        total_specs_count = int(parts[1])
                    except ValueError:
                        pass # Keep counts at 0 if parsing fails

                # ✅ Message construction based on operation success, not validation score
                if not validation_op_successful and "error" in validation_data:
                    final_message = validation_data.get('error', 'Validation process error.')
                elif validation_op_successful:
                    # ✅ Even with low scores (0%, 20%), this is a successful validation
                    final_message = f"Specifications matched for {prod_name_for_message}."
                    if final_score_for_message is not None: 
                        final_message += f" (Score: {final_score_for_message}%)"
                    # Report matched/total based on specifications_match
                    final_message += f" [{matched_specs_count}/{total_specs_count} spec(s) matched"
                    if unmatched_specs_count > 0 and unmatched_specs_count == (total_specs_count - matched_specs_count):
                        final_message += f", {unmatched_specs_count} unmatched]"
                    elif unmatched_specs_count > 0:
                        final_message += f", {unmatched_specs_count} explicitly unmatched]"
                    else:
                        final_message += "]"
                else:
                    final_message = "Validation failed or LLM response was inconclusive."
                
                response_payload["message"] = final_message
                
                # Add product metadata
                response_payload["product_name"] = product_to_validate.get('name', 'Unknown Product')
                response_payload["specifications_count"] = len(product_to_validate.get('specifications', []))
                
                # ✅ Ensure all validation data fields are preserved
                # validation_data already contains: valid, validation_score, product_name_found, 
                # specifications_match, matched_specifications, unmatched_specifications,
                # any_manufacturer_found, found_manufacturers, unmatched_manufacturers,
                # summary, response, elapsed_time, tokens, model_used
                
                # Log audit trail
                await log_api_success(
                    request=request,
                    action="SmartSearchValidate",
                    entity_type="PDF",
                    entity_id=pdf_url,
                    metadata={
                        "PDF_Name": safe_filename,
                        "product": product_to_validate.get('name'),
                        "score": validation_data.get("validation_score"),
                        "final_result": final_message,
                        "model": val_model_used,
                        "tokens": f"{val_prompt_tokens}/{val_completion_tokens}"
                    },
                    cost=val_total_cost,
                    start_time=start_time
                )
                
                logger.info(f"✅ Validation complete - Score: {final_score_for_message}%, Success: {validation_op_successful}")
                return response_payload
                
        # These except blocks now correctly pair with the inner 'try:'
        except HTTPException as http_exc:
            await log_api_failure(
                request=request,
                action="SmartSearchValidate",
                entity_type="PDF",
                entity_id=pdf_url,
                error=str(http_exc.detail),
                start_time=start_time
            )
            raise http_exc
        except httpx.RequestError as req_err:
            await log_api_failure(
                request=request,
                action="SmartSearchValidate",
                entity_type="PDF",
                entity_id=pdf_url,
                error=str(req_err),
                start_time=start_time
            )
            logger.error(f"HTTPX Request error fetching PDF from {pdf_url}: {req_err}")
            raise HTTPException(status_code=502, detail=f"Network error fetching PDF: {req_err}")
        except Exception as e_inner:
            logger.error(f"Inner error during validation for URL {pdf_url}: {e_inner}", exc_info=True)
            await log_api_failure(
                request=request,
                action="SmartSearchValidate",
                entity_type="PDF",
                entity_id=pdf_url,
                error=str(e_inner),
                start_time=start_time
            )
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": f"An inner processing error occurred: {str(e_inner)}",
                    "validation_score": None,
                    "product_name": product_to_validate.get('name', 'Unknown') if product_to_validate else 'Unknown',
                    "specifications_count": None,
                    "summary": None,
                    "unmatched_specifications": None,
                }
            )
    # This is the outermost except block for the entire endpoint functiondf 
    except Exception as e_outer:
        logger.error(f"Outer error during validation for URL {pdf_url}: {e_outer}", exc_info=True)
        await log_api_failure(
            request=request,
            action="SmartSearchValidate",
            entity_type="PDF",
            entity_id=pdf_url,
            error=str(e_outer),
            start_time=start_time
        )
        return JSONResponse(
            status_code=500, 
            content={
                "success": False,
                "message": f"An unexpected overall error occurred: {str(e_outer)}",
                "validation_score": None,
                "product_name": product_to_validate.get('name', 'Unknown') if product_to_validate else 'Unknown',
                "specifications_count": None,
                "summary": None,
                "unmatched_specifications": None,
            }
        )


@app.post("/api/smart-validate-specs")
@limiter.limit("10/minute")
async def smart_validate_specifications(
    request: Request,
    pdf_file: UploadFile,
    product_data_json: str = Form(...),
    x_session_id: Optional[str] = Header(None),
    current_user: User = Depends(get_current_active_user)
):
    logger.info(f"Received request for /api/smart-validate-specs with SessionID: {x_session_id}")
    product_to_validate = None
    temp_pdf_path_for_logging = pdf_file.filename
    filesize_for_logging = 0
    process_start = time.time()

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Create a safe filename for the temporary file
                safe_filename = f"validation_{x_session_id or 'temp'}.pdf"
                temp_pdf_path = os.path.join(temp_dir, safe_filename)
                temp_pdf_path_for_logging = safe_filename

                # Save the uploaded file to temp location
                filesize_for_logging = 0
                async with aiofiles.open(temp_pdf_path, "wb") as buffer:
                    while chunk := await pdf_file.read(4096):  # Read in chunks
                        await buffer.write(chunk)
                        filesize_for_logging += len(chunk)
                        
                        # Prevent extremely large uploads (> 100MB)
                        if filesize_for_logging > 100 * 1024 * 1024:
                            raise HTTPException(
                                status_code=413, 
                                detail="PDF file too large (>100MB)"
                            )

                # Read the file back for validation
                async with aiofiles.open(temp_pdf_path, "rb") as f:
                    pdf_content_bytes = await f.read()

                if filesize_for_logging == 0:
                    logger.warning("Uploaded PDF is empty")
                    return JSONResponse(
                        status_code=400,
                        content={
                            "success": False,
                            "message": "Uploaded PDF was empty",
                            "validation_score": 0,
                            "product_name": "Unknown",
                            "specifications_count": None,
                            "summary": None,
                            "unmatched_specifications": None
                        }
                    )

                # Parse product data
                try:
                    product_to_validate = json.loads(product_data_json)
                    if not isinstance(product_to_validate, dict) or 'name' not in product_to_validate:
                        raise HTTPException(
                            status_code=422, 
                            detail="Invalid product data structure"
                        )
                except json.JSONDecodeError as e_json:
                    raise HTTPException(
                        status_code=422, 
                        detail=f"Invalid product data JSON: {e_json}"
                    )

                # Import validation function
                try:
                    from validate_with_llm import validate_pdf_with_llm
                except ImportError:
                    from .validate_with_llm import validate_pdf_with_llm

                # Run validation
                validation_op_successful, validation_data = await run_in_threadpool(
                    validate_pdf_with_llm,
                    pdf_path=temp_pdf_path,
                    product_data=product_to_validate
                )

                # Calculate token costs
                val_prompt_tokens = validation_data.get("prompt_tokens", 0)
                val_completion_tokens = validation_data.get("completion_tokens", 0)
                val_model_used = validation_data.get("model_used", "gemini-2.0-flash")
                
                if isinstance(val_model_used, dict):
                    logger.warning(f"val_model_used is a dict: {val_model_used}, using default model")
                    val_model_used = "gemini-2.0-flash"

                val_token_cost = get_token_cost(val_model_used, val_prompt_tokens, val_completion_tokens)
                val_fixed_cost = 0.0
                val_total_cost = val_token_cost + val_fixed_cost
                
                # Log usage
                current_time = datetime.datetime.now(datetime.timezone.utc)
                await run_in_threadpool(
                    log_usage_to_csv,
                    timestamp=current_time,
                    session_id=x_session_id,
                    filename=temp_pdf_path_for_logging,
                    filesize=filesize_for_logging,
                    request_type="smart_llm_validation",
                    model_used=val_model_used,
                    input_tokens=val_prompt_tokens,
                    output_tokens=val_completion_tokens,
                    grounded_searches=0,
                    token_cost=val_token_cost,
                    fixed_cost=val_fixed_cost,
                    total_cost=val_total_cost
                )
                logger.info(f"Validation - File: {temp_pdf_path_for_logging}, SessionID: {x_session_id}, Model: {val_model_used}, Input: {val_prompt_tokens}, Output: {val_completion_tokens}, TokenCost: ${val_token_cost:.6f}, TotalCost: ${val_total_cost:.6f}")
                
                # ============================================================================
                # ✅ CRITICAL FIX: Response Construction (same as validate-specs)
                # ============================================================================
                response_payload = validation_data.copy()
                
                # ✅ FIX: success based on operation success, not validation score
                response_payload["success"] = validation_op_successful
                response_payload["validation_score"] = validation_data.get("validation_score", 0)
                
                # Construct message
                prod_name_for_message = product_to_validate.get('name', 'Unknown Product')
                final_score_for_message = validation_data.get("validation_score", 0)
                unmatched_specs_count = len(validation_data.get("unmatched_specifications", []))
                specs_match_str = validation_data.get("specifications_match", "0/0")
                
                matched_specs_count = 0
                total_specs_count = 0
                if '/' in specs_match_str:
                    try:
                        parts = specs_match_str.split('/')
                        matched_specs_count = int(parts[0])
                        total_specs_count = int(parts[1])
                    except ValueError:
                        pass

                # ✅ Message based on operation success, not score
                if not validation_op_successful and "error" in validation_data:
                    final_message = validation_data.get('error', 'Validation process error.')
                elif validation_op_successful:
                    # ✅ Even low scores are successful validations
                    final_message = f"Specifications matched for {prod_name_for_message}."
                    if final_score_for_message is not None: 
                        final_message += f" (Score: {final_score_for_message}%)"
                    final_message += f" [{matched_specs_count}/{total_specs_count} spec(s) matched"
                    if unmatched_specs_count > 0 and unmatched_specs_count == (total_specs_count - matched_specs_count):
                        final_message += f", {unmatched_specs_count} unmatched]"
                    elif unmatched_specs_count > 0:
                        final_message += f", {unmatched_specs_count} explicitly unmatched]"
                    else:
                        final_message += "]"
                else:
                    final_message = "Validation failed or LLM response was inconclusive."
                
                response_payload["message"] = final_message
                response_payload["product_name"] = prod_name_for_message
                response_payload["specifications_count"] = len(product_to_validate.get('specifications', []))

                # Log audit trail
                await log_api_success(
                    request=request,
                    action="SmartValidate",
                    entity_type="Product",
                    entity_id=prod_name_for_message,
                    metadata={
                        "file_name": safe_filename,
                        "product_name": prod_name_for_message,
                        "validation_score": validation_data.get("validation_score"),
                        "matched_specifications": validation_data.get("specifications_match"),
                        "unmatched_count": unmatched_specs_count,
                        "model_used": val_model_used,
                        "input_tokens": val_prompt_tokens,
                        "output_tokens": val_completion_tokens
                    },
                    cost=val_total_cost,
                    start_time=process_start
                )

                logger.info(f"✅ Smart validation complete - Score: {final_score_for_message}%, Success: {validation_op_successful}")
                return response_payload

            except HTTPException as http_exc:
                await log_api_failure(
                    request=request,
                    action="SmartValidate",
                    entity_type="Product",
                    entity_id=product_to_validate.get('name', 'Unknown') if product_to_validate else 'Unknown',
                    error=str(http_exc.detail),
                    start_time=process_start
                )
                raise http_exc
            except Exception as e_inner:
                logger.error(f"Inner error during smart validation: {e_inner}", exc_info=True)
                await log_api_failure(
                    request=request,
                    action="SmartValidate",
                    entity_type="Product",
                    entity_id=product_to_validate.get('name', 'Unknown') if product_to_validate else 'Unknown',
                    error=str(e_inner),
                    start_time=process_start
                )
                return JSONResponse(
                    status_code=500,
                    content={
                        "success": False, 
                        "message": f"An inner processing error occurred: {str(e_inner)}",
                        "validation_score": None,
                        "product_name": product_to_validate.get('name', 'Unknown') if product_to_validate else 'Unknown',
                        "specifications_count": None, 
                        "summary": None, 
                        "unmatched_specifications": None,
                    }
                )
    except Exception as e_outer:
        logger.error(f"Outer error during smart validation: {e_outer}", exc_info=True)
        await log_api_failure(
            request=request,
            action="SmartValidate",
            entity_type="Product",
            entity_id=product_to_validate.get('name', 'Unknown') if product_to_validate else 'Unknown',
            error=str(e_outer),
            start_time=process_start
        )
        return JSONResponse(
            status_code=500, 
            content={
                "success": False, 
                "message": f"An unexpected overall error occurred: {str(e_outer)}",
                "validation_score": None,
                "product_name": product_to_validate.get('name', 'Unknown') if product_to_validate else 'Unknown',
                "specifications_count": None, 
                "summary": None, 
                "unmatched_specifications": None,
            }
        )

@app.post("/api/extract-pds-links")
async def extract_pds_links_endpoint(
    request: Request,
    body: PDFLinkExtractionRequest,
    current_user: User = Depends(get_current_active_user),
    x_session_id: Optional[str] = Header(None)
):
    process_start = time.time()
    # Access refresh and product_name from body, not request!
    refresh_status = "with cache refresh" if body.refresh else "using cache if available"
    logger.info(f"Received PDS link extraction request for product: {body.product_name}, SessionID: {x_session_id}, {refresh_status}")
    try:
        product_data_dict = body.dict()
        # Remove refresh from product_data_dict as it's not part of the product data for LLM
        refresh_requested = product_data_dict.pop('refresh', False)

        # extract_pdf_links now returns a dict: {"results": [], "prompt_tokens": X, ...}
        extraction_response_dict = await run_in_threadpool(extract_pdf_links, product_data_dict, refresh_requested)
        
        results = extraction_response_dict.get("results", [])
        pds_prompt_tokens = extraction_response_dict.get("prompt_tokens", 0)
        pds_completion_tokens = extraction_response_dict.get("completion_tokens", 0)
        pds_model_used = extraction_response_dict.get("model_used", MODEL_PRICING["unknown_model"])  # Default if not in result

        # Calculate cost for PDS link extraction
        pds_token_cost = get_token_cost(pds_model_used, pds_prompt_tokens, pds_completion_tokens)
        pds_fixed_cost = GROUNDED_SEARCH_FIXED_COST_PER_REQUEST  # One grounded search operation
        pds_total_cost = pds_token_cost + pds_fixed_cost

        # Log usage to CSV
        current_time = datetime.datetime.now(datetime.timezone.utc)
        filename_for_log = f"pds_extraction_for_{product_data_dict.get('product_name', 'unknown_product')}"

        await run_in_threadpool(
            log_usage_to_csv,
            timestamp=current_time,
            session_id=x_session_id,
            filename=filename_for_log,
            filesize=0,
            request_type="grounded_search_llm",
            model_used=pds_model_used,
            input_tokens=pds_prompt_tokens,
            output_tokens=pds_completion_tokens,
            grounded_searches=1,
            token_cost=pds_token_cost,
            fixed_cost=pds_fixed_cost,
            total_cost=pds_total_cost
        )

        # --- AUDIT LOG (Success) ---
        await log_api_success(
            request=request,
            action="SmartSearch",
            entity_type="Product",
            entity_id=body.product_name,
            metadata={
                "pdf_name":filename_for_log,
                "product_name": body.product_name,
                "refresh": refresh_requested,
                "results_count": len(results),
                "model_used": pds_model_used,
                "input_tokens": pds_prompt_tokens,
                "output_tokens": pds_completion_tokens
            },
            cost=pds_total_cost,
            start_time=process_start
        )

        refresh_msg = " (REFRESHED)" if refresh_requested else ""
        logger.info(f"PDS Links{refresh_msg} - Product: {product_data_dict.get('product_name', 'N/A')}, SessionID: {x_session_id}, Model: {pds_model_used}, Input: {pds_prompt_tokens}, Output: {pds_completion_tokens}, TokenCost: ${pds_token_cost:.6f}, FixedCost: ${pds_fixed_cost:.6f}, TotalCost: ${pds_total_cost:.6f}")

        logger.info(f"PDF extraction results (count): {len(results)}")
        return {"results": results}

    except Exception as e:
        # --- AUDIT LOG (Failure) ---
        await log_api_failure(
            request=request,
            action="SmartSearch",
            entity_type="Product",
            entity_id=getattr(body, "product_name", None),
            error=str(e),
            start_time=process_start
        )
        logger.error(f"Error during PDF link extraction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during PDF link extraction: {str(e)}")

@app.post("/api/add-validated-pdfs")
async def add_validated_pdfs(
    request_data: AddValidatedPDFsRequest,
    request: Request,
    client: httpx.AsyncClient = Depends(lambda: app.state.httpx_client),
    x_session_id: Optional[str] = Header(None),
    current_user: User = Depends(get_current_active_user)
):
    """
    Validate PDFs (if not already validated) and merge validation reports as first page.
    Returns processed PDFs ready for download.
    """
    logger.info(
        f"Received request for /api/add-validated-pdfs with {len(request_data.new_pdfs)} new PDFs "
        f"and {len(request_data.already_validated_pdfs)} already validated PDFs, SessionID: {x_session_id}"
    )

    processed_pdfs = []
    total_validation_cost = 0.0
    audit_start_time = time.time()

    # ---------- Process new PDFs ----------
    for i, pdf_item in enumerate(request_data.new_pdfs):
        try:
            logger.info(f"Processing new PDF {i+1}/{len(request_data.new_pdfs)}: {pdf_item.pdf_url}")

            # Import validator here if needed
            try:
                from validate_with_llm import validate_pdf_with_llm
            except ImportError:
                from .validate_with_llm import validate_pdf_with_llm

            with tempfile.TemporaryDirectory() as temp_dir:
                # Download PDF
                try:
                    url_hash = hashlib.md5(pdf_item.pdf_url.encode()).hexdigest()
                    safe_filename = f"{url_hash}.pdf"
                except Exception:
                    safe_filename = f"temp_{i}.pdf"

                temp_pdf_path = os.path.join(temp_dir, safe_filename)

                max_retries = 3
                retry_delay = 1
                pdf_content_bytes = None
                download_successful = False

                for attempt in range(max_retries):
                    try:
                        timeout_config = httpx.Timeout(
                            connect=30.0, read=180.0, write=30.0, pool=240.0
                        )
                        async with client.stream(
                            "GET", pdf_item.pdf_url,
                            headers={"User-Agent": "Mozilla/5.0"},
                            follow_redirects=True, timeout=timeout_config
                        ) as response:
                            response.raise_for_status()
                            total_size = 0
                            async with aiofiles.open(temp_pdf_path, "wb") as buffer:
                                async for chunk in response.aiter_bytes(chunk_size=4096):
                                    await buffer.write(chunk)
                                    total_size += len(chunk)
                                    if total_size > 100 * 1024 * 1024:
                                        raise HTTPException(status_code=413, detail="PDF file too large (>100MB)")
                            async with aiofiles.open(temp_pdf_path, "rb") as f:
                                pdf_content_bytes = await f.read()
                            download_successful = True
                        break
                    except (httpx.RequestError, httpx.TimeoutException, httpx.ReadTimeout) as e:
                        await log_api_failure(
                            request=request,
                            action="SmartSearchValidate",
                            entity_type="BatchPDFValidation",
                            entity_id=None,
                            error=str(e),
                            start_time=audit_start_time
                        )
                        if attempt == max_retries - 1:
                            logger.error(f"Failed to download PDF after {max_retries} attempts: {e}")
                        else:
                            logger.warning(f"PDF download attempt {attempt + 1} failed, retrying in {retry_delay}s: {e}")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2
                    except Exception as e:
                        await log_api_failure(
                            request=request,
                            action="SmartSearchValidate",
                            entity_type="BatchPDFValidation",
                            entity_id=None,
                            error=str(e),
                            start_time=audit_start_time
                        )
                        logger.error(f"Unexpected error downloading PDF on attempt {attempt + 1}: {e}")
                        if attempt == max_retries - 1:
                            break

                if not download_successful or not pdf_content_bytes:
                    logger.error(f"Unable to download PDF for validation: {pdf_item.pdf_url}")
                    processed_pdfs.append({
                        "original_url": pdf_item.pdf_url,
                        "filename": pdf_item.filename or f"failed_download_{i}.pdf",
                        "product_name": pdf_item.product_data.get('name', 'Unknown'),
                        "validation_score": 0,
                        "pdf_data": None,
                        "validation_summary": "Failed to download PDF for validation - Looks like this PDF is protected and can't be accessed through our system. No worries — just download it to your device and upload it directly.",
                        "success": False
                    })
                    continue

                filesize = len(pdf_content_bytes) if pdf_content_bytes else 0
                if filesize == 0:
                    logger.warning(f"Downloaded PDF is empty for URL: {pdf_item.pdf_url}")
                    processed_pdfs.append({
                        "original_url": pdf_item.pdf_url,
                        "filename": pdf_item.filename or f"empty_pdf_{i}.pdf",
                        "product_name": pdf_item.product_data.get('name', 'Unknown'),
                        "validation_score": 0,
                        "pdf_data": None,
                        "validation_summary": "Downloaded PDF was empty",
                        "success": False
                    })
                    continue

                # Validation
                validation_successful, validation_data = await run_in_threadpool(
                    validate_pdf_with_llm,
                    pdf_path=temp_pdf_path,
                    product_data=pdf_item.product_data
                )

                if not validation_successful:
                    logger.warning(f"Validation failed for PDF: {pdf_item.pdf_url}")
                    validation_data = {
                        "validation_score": 0,
                        "summary": "Validation failed - unable to process PDF",
                        "matched_specifications": [],
                        "unmatched_specifications": [],
                        "found_manufacturers": [],
                        "unmatched_manufacturers": [],
                        "warnings": ["Validation process failed"],
                        "valid": "No"
                    }

                product_name = pdf_item.product_data.get('name', 'Unknown Product')
                merged_pdf_bytes = await run_in_threadpool(
                    generate_validation_report_and_merge,
                    validation_data,
                    product_name,
                    original_pdf_url=None,
                    original_pdf_bytes=pdf_content_bytes
                )

                if not merged_pdf_bytes:
                    logger.error(f"Failed to generate merged PDF for: {pdf_item.pdf_url}")
                    continue

                pdf_base64 = base64.b64encode(merged_pdf_bytes).decode('utf-8')
                val_prompt_tokens = validation_data.get("prompt_tokens", 0)
                val_completion_tokens = validation_data.get("completion_tokens", 0)
                val_model_used = validation_data.get("model_used", "gemini-2.0-flash")
                if isinstance(val_model_used, dict):
                    logger.warning(f"val_model_used is a dict: {val_model_used}, using default model")
                    val_model_used = "gemini-2.0-flash"
                val_token_cost = get_token_cost(val_model_used, val_prompt_tokens, val_completion_tokens)
                total_validation_cost += val_token_cost

                current_time = datetime.datetime.now(datetime.timezone.utc)
                await run_in_threadpool(
                    log_usage_to_csv,
                    timestamp=current_time,
                    session_id=x_session_id,
                    filename=pdf_item.filename or safe_filename,
                    filesize=filesize,
                    request_type="pdf_validation_with_report",
                    model_used=val_model_used,
                    input_tokens=val_prompt_tokens,
                    output_tokens=val_completion_tokens,
                    grounded_searches=0,
                    token_cost=val_token_cost,
                    fixed_cost=0.0,
                    total_cost=val_token_cost
                )

                processed_pdfs.append({
                    "original_url": pdf_item.pdf_url,
                    "filename": pdf_item.filename or f"{product_name}_validated.pdf",
                    "product_name": product_name,
                    "validation_score": validation_data.get("validation_score", 0),
                    "pdf_data": pdf_base64,
                    "validation_summary": validation_data.get("summary", ""),
                    "success": True
                })

                logger.info(f"Successfully processed new PDF {i+1}: {product_name}, Score: {validation_data.get('validation_score', 0)}%")

        except Exception as e:
            await log_api_failure(
                request=request,
                action="SmartSearchValidate",
                entity_type="BatchPDFValidation",
                entity_id=None,
                error=str(e),
                start_time=audit_start_time
            )
            logger.error(f"Error processing new PDF {i+1} ({pdf_item.pdf_url}): {str(e)}", exc_info=True)
            processed_pdfs.append({
                "original_url": pdf_item.pdf_url,
                "filename": pdf_item.filename or f"error_{i}.pdf",
                "product_name": pdf_item.product_data.get('name', 'Unknown'),
                "validation_score": 0,
                "pdf_data": None,
                "validation_summary": f"Error processing PDF: {str(e)}",
                "success": False
            })

    # ---------- Process already validated PDFs ----------
    for i, pdf_item in enumerate(request_data.already_validated_pdfs):
        try:
            logger.info(f"Processing already validated PDF {i+1}/{len(request_data.already_validated_pdfs)}: {pdf_item.pdf_url}")
            validation_data = pdf_item.validation_data

            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    url_hash = hashlib.md5(pdf_item.pdf_url.encode()).hexdigest()
                    safe_filename = f"{url_hash}.pdf"
                except Exception:
                    safe_filename = f"temp_validated_{i}.pdf"

                temp_pdf_path = os.path.join(temp_dir, safe_filename)
                max_retries = 3
                retry_delay = 1
                pdf_content_bytes = None
                download_successful = False

                for attempt in range(max_retries):
                    try:
                        timeout_config = httpx.Timeout(
                            connect=60.0, read=300.0, write=60.0, pool=360.0
                        )
                        async with client.stream(
                            "GET", pdf_item.pdf_url,
                            headers={"User-Agent": "Mozilla/5.0"},
                            follow_redirects=True, timeout=timeout_config
                        ) as response:
                            response.raise_for_status()
                            total_size = 0
                            async with aiofiles.open(temp_pdf_path, "wb") as buffer:
                                async for chunk in response.aiter_bytes(chunk_size=4096):
                                    await buffer.write(chunk)
                                    total_size += len(chunk)
                                    if total_size > 100 * 1024 * 1024:
                                        raise HTTPException(status_code=413, detail="PDF file too large (>100MB)")
                            async with aiofiles.open(temp_pdf_path, "rb") as f:
                                pdf_content_bytes = await f.read()
                            download_successful = True
                        break
                    except (httpx.RequestError, httpx.TimeoutException, httpx.ReadTimeout) as e:
                        if attempt == max_retries - 1:
                            await log_api_failure(
                                request=request,
                                action="SmartSearchValidate",
                                entity_type="BatchPDFValidation",
                                entity_id=None,
                                error=str(e),
                                start_time=audit_start_time
                            )
                            logger.error(f"Failed to download already validated PDF after {max_retries} attempts: {e}")
                        else:
                            await log_api_failure(
                                request=request,
                                action="SmartSearchValidate",
                                entity_type="BatchPDFValidation",
                                entity_id=None,
                                error=str(e),
                                start_time=audit_start_time
                            )
                            logger.warning(f"Already validated PDF download attempt {attempt + 1} failed, retrying in {retry_delay}s: {e}")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2
                    except Exception as e:
                        await log_api_failure(
                            request=request,
                            action="SmartSearchValidate",
                            entity_type="BatchPDFValidation",
                            entity_id=None,
                            error=str(e),
                            start_time=audit_start_time
                        )
                        logger.error(f"Unexpected error downloading already validated PDF on attempt {attempt + 1}: {e}")
                        if attempt == max_retries - 1:
                            break

                if not download_successful or not pdf_content_bytes:
                    logger.error(f"Unable to download already validated PDF: {pdf_item.pdf_url}")
                    processed_pdfs.append({
                        "original_url": pdf_item.pdf_url,
                        "filename": pdf_item.filename or f"failed_download_validated_{i}.pdf",
                        "product_name": pdf_item.product_data.get('name', 'Unknown'),
                        "validation_score": 0,
                        "pdf_data": None,
                        "validation_summary": "Failed to download already validated PDF - Looks like this PDF is protected and can't be accessed through our system. No worries — just download it to your device and upload it directly.",
                        "success": False
                    })
                    continue

                filesize = len(pdf_content_bytes) if pdf_content_bytes else 0
                if filesize == 0:
                    logger.warning(f"Downloaded already validated PDF is empty for URL: {pdf_item.pdf_url}")
                    processed_pdfs.append({
                        "original_url": pdf_item.pdf_url,
                        "filename": pdf_item.filename or f"empty_validated_{i}.pdf",
                        "product_name": pdf_item.product_data.get('name', 'Unknown'),
                        "validation_score": 0,
                        "pdf_data": None,
                        "validation_summary": "Downloaded already validated PDF was empty",
                        "success": False
                    })
                    continue

                product_name = pdf_item.product_data.get('name', 'Unknown Product')
                merged_pdf_bytes = await run_in_threadpool(
                    generate_validation_report_and_merge,
                    validation_data,
                    product_name,
                    original_pdf_url=None,
                    original_pdf_bytes=pdf_content_bytes
                )

                if not merged_pdf_bytes:
                    logger.error(f"Failed to generate merged PDF for already validated: {pdf_item.pdf_url}")
                    processed_pdfs.append({
                        "original_url": pdf_item.pdf_url,
                        "filename": pdf_item.filename or f"merge_failed_validated_{i}.pdf",
                        "product_name": product_name,
                        "validation_score": validation_data.get("validation_score", 0),
                        "pdf_data": None,
                        "validation_summary": "Failed to merge validation report with PDF",
                        "success": False
                    })
                    continue

                pdf_base64 = base64.b64encode(merged_pdf_bytes).decode('utf-8')
                logger.info(f"No validation cost - using existing validation data for: {pdf_item.pdf_url}")
                processed_pdfs.append({
                    "original_url": pdf_item.pdf_url,
                    "filename": pdf_item.filename or f"{product_name}_validated.pdf",
                    "product_name": product_name,
                    "validation_score": validation_data.get("validation_score", 0),
                    "pdf_data": pdf_base64,
                    "validation_summary": validation_data.get("summary", ""),
                    "success": True
                })
                logger.info(f"Successfully processed already validated PDF {i+1}: {product_name}, Score: {validation_data.get('validation_score', 0)}% (no re-validation)")

        except Exception as e:
            await log_api_failure(
                request=request,
                action="SmartSearchValidate",
                entity_type="BatchPDFValidation",
                entity_id=None,
                error=str(e),
                start_time=audit_start_time
            )
            logger.error(f"Error processing already validated PDF {i+1} ({pdf_item.pdf_url}): {str(e)}", exc_info=True)
            processed_pdfs.append({
                "original_url": pdf_item.pdf_url,
                "filename": pdf_item.filename or f"error_validated_{i}.pdf",
                "product_name": pdf_item.product_data.get('name', 'Unknown'),
                "validation_score": 0,
                "pdf_data": None,
                "validation_summary": f"Error processing PDF: {str(e)}",
                "success": False
            })

    logger.info(
        f"Completed processing {len(request_data.new_pdfs)} new PDFs and {len(request_data.already_validated_pdfs)} already validated PDFs. "
        f"Successfully processed: {sum(1 for p in processed_pdfs if p['success'])}"
    )
    await log_api_success(
        request=request,
        action="SmartSearchValidate",
        entity_type="BatchPDFValidation",
        entity_id=None,
        metadata={
            "new_pdfs_count": len(request_data.new_pdfs),
            "already_validated_pdfs_count": len(request_data.already_validated_pdfs),
            "processed_total": len(processed_pdfs),
            "successful_count": sum(1 for p in processed_pdfs if p['success']),
            "total_validation_cost": round(total_validation_cost, 6),
            "session_id": x_session_id
        },
        cost=total_validation_cost,
        start_time=audit_start_time
    )

    return {
        "processed_pdfs": processed_pdfs,
        "total_processed": len(processed_pdfs),
        "successful_count": sum(1 for p in processed_pdfs if p['success']),
        "total_validation_cost": round(total_validation_cost, 6)
    }



# --- Include Routers ---
app.include_router(pdf_markers_router, prefix="/api/pdf-tools")
app.include_router(trim_pdf_router, prefix="/api/pdf-tools")

@app.get("/api/usage-summary")
async def get_usage_summary(session_id: Optional[str] = Query(None)):
    logger.info(f"Received request for /api/usage-summary, SessionID: {session_id}")
    
    # Default error response that matches the frontend's expected structure
    error_response = {
        "all_time_total_token_cost": 0,
        "all_time_total_fixed_cost": 0,
        "all_time_total_cost": 0,
        "all_time_total_requests": 0,
        "by_model": {},
        "by_request_type": {},
        "session_id_filter": session_id
    }
    
    # File existence check with better error handling
    try:
        if not LOG_FILE_PATH.is_file():
            logger.warning(f"Usage log file {LOG_FILE_PATH} not found.")
            error_response["error"] = "Usage log file not found"
            return JSONResponse(content=error_response, status_code=404)
        
        if os.path.getsize(LOG_FILE_PATH) == 0:
            logger.warning(f"Usage log file {LOG_FILE_PATH} is empty.")
            error_response["error"] = "Usage log file is empty"
            return JSONResponse(content=error_response, status_code=404)
    except Exception as e:
        logger.error(f"Error checking log file: {e}", exc_info=True)
        error_response["error"] = f"Error checking log file: {str(e)}"
        return JSONResponse(content=error_response, status_code=500)

    # Initialize summary structure
    summary = {
        "total_requests": 0,
        "total_token_cost_usd": 0.0,
        "total_fixed_cost_usd": 0.0,
        "grand_total_estimated_cost_usd": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_grounded_searches": 0,
        "cost_by_model": defaultdict(lambda: {"requests": 0, "input_tokens": 0, "output_tokens": 0, "token_cost": 0.0, "fixed_cost": 0.0, "total_cost": 0.0, "grounded_searches": 0}),
        "cost_by_date": defaultdict(lambda: {"requests": 0, "token_cost": 0.0, "fixed_cost": 0.0, "total_cost": 0.0, "grounded_searches": 0}),
        "cost_by_request_type": defaultdict(lambda: {"requests": 0, "input_tokens": 0, "output_tokens": 0, "token_cost": 0.0, "fixed_cost": 0.0, "total_cost": 0.0, "grounded_searches": 0}),
        "errors": []
    }

    # Read CSV data
    try:
        logger.info("Reading CSV data from log file")
        
        def read_csv_data():
            records = []
            try:
                with csv_lock:
                    with open(LOG_FILE_PATH, 'r', newline='', encoding='utf-8') as csvfile:
                        reader = csv.DictReader(csvfile)
                        
                        # Check for header mismatch
                        if not reader.fieldnames:
                            return None, "CSV has no field names"
                        
                        # Log the actual fieldnames for debugging
                        logger.debug(f"CSV fieldnames: {reader.fieldnames}")
                        
                        # Create a mapping from current expected column names to possible alternatives/legacy formats
                        column_mapping = {
                            "Timestamp": ["Timestamp", "timestamp", "Time", "time", "Date Time"],
                            "Session ID": ["Session ID", "SessionID", "session_id", "Session", "session"],
                            "Filename": ["Filename", "filename", "File", "file", "PDF"],
                            "Filesize (Bytes)": ["Filesize (Bytes)", "Filesize", "filesize", "Size", "size"],
                            "Request Type": ["Request Type", "RequestType", "request_type", "Type", "type"],
                            "Model Used": ["Model Used", "ModelUsed", "model_used", "Model", "model"],
                            "Input Tokens": ["Input Tokens", "InputTokens", "input_tokens", "Prompt Tokens"],
                            "Output Tokens": ["Output Tokens", "OutputTokens", "output_tokens", "Completion Tokens"],
                            "Grounded Searches Performed": ["Grounded Searches Performed", "GroundedSearches", "grounded_searches", "Searches"],
                            "Token Cost USD": ["Token Cost USD", "TokenCost", "token_cost", "Token Cost"],
                            "Fixed Cost USD": ["Fixed Cost USD", "FixedCost", "fixed_cost", "Fixed Cost"],
                            "Total Estimated Cost USD": ["Total Estimated Cost USD", "TotalCost", "total_cost", "Total Cost", "Estimated Cost"]
                        }
                        
                        # Create field name translator dict based on what's available in the CSV
                        field_translator = {}
                        existing_column_set = set(reader.fieldnames)
                        missing_required_fields = []
                        
                        for expected_name, alternatives in column_mapping.items():
                            # Check if any of the alternatives are present
                            found = False
                            for alt in alternatives:
                                if alt in existing_column_set:
                                    field_translator[expected_name] = alt
                                    found = True
                                    break
                            
                            if not found:
                                # For backward compatibility, supply default values for missing fields
                                if expected_name in ["Session ID", "Request Type"]:
                                    # These are absolutely required - can't proceed without them
                                    missing_required_fields.append(expected_name)
                                else:
                                    # For other missing fields, we'll supply defaults in the code
                                    logger.warning(f"Missing column '{expected_name}' in CSV - will use defaults")
                        
                        if missing_required_fields:
                            # Handle missing required fields by creating a new CSV with proper headers
                            logger.warning(f"CSV is missing required columns: {missing_required_fields}. This may be an old format.")
                            logger.warning(f"For best results, you may want to rename your existing usage_log.csv and let the app create a new one.")
                            # Don't fail completely - we'll try to proceed with empty values for these fields
                            pass
                        
                        for row in reader:
                            # Create a standardized row with expected column names
                            standardized_row = {}
                            
                            for expected_name, alternatives in column_mapping.items():
                                # Use the translator to find the right field name if we have it
                                if expected_name in field_translator:
                                    csv_field_name = field_translator[expected_name]
                                    standardized_row[expected_name] = row[csv_field_name]
                                else:
                                    # Supply defaults for missing fields based on their type
                                    if expected_name == "Session ID":
                                        standardized_row[expected_name] = "legacy_data"
                                    elif expected_name == "Request Type":
                                        standardized_row[expected_name] = "standard_llm_extraction"
                                    elif expected_name in ["Grounded Searches Performed"]:
                                        standardized_row[expected_name] = "0"
                                    elif expected_name in ["Token Cost USD", "Fixed Cost USD"]:
                                        # Try to estimate from Estimated Cost if available
                                        if "Estimated Cost (USD)" in row:
                                            estimated_cost = float(row["Estimated Cost (USD)"])
                                            if expected_name == "Token Cost USD":
                                                standardized_row[expected_name] = str(estimated_cost)
                                            else:  # Fixed Cost USD
                                                standardized_row[expected_name] = "0.0" 
                                        else:
                                            standardized_row[expected_name] = "0.0"
                                    elif expected_name == "Total Estimated Cost USD":
                                        # Try different field names that might contain the total cost
                                        if "Estimated Cost (USD)" in row:
                                            standardized_row[expected_name] = row["Estimated Cost (USD)"]
                                        elif "Estimated Cost" in row:
                                            standardized_row[expected_name] = row["Estimated Cost"]
                                        else:
                                            standardized_row[expected_name] = "0.0"
                                    else:
                                        standardized_row[expected_name] = ""
                            
                            # Filter by session_id if provided
                            if session_id and standardized_row.get("Session ID") != session_id:
                                continue
                                
                            records.append(standardized_row)
                
                return records, None
            except Exception as e:
                logger.error(f"Error in read_csv_data: {e}", exc_info=True)
                return None, f"Error reading CSV: {str(e)}"
        
        log_records, csv_error = await run_in_threadpool(read_csv_data)
        
        if csv_error:
            logger.error(f"CSV read error: {csv_error}")
            error_response["error"] = csv_error
            return JSONResponse(content=error_response, status_code=500)
            
        if not log_records:
            logger.warning("No log records found (or none matching the session_id filter)")
            # Return empty summary with the correct structure
            frontend_summary = {
                "all_time_total_token_cost": 0,
                "all_time_total_fixed_cost": 0, 
                "all_time_total_cost": 0,
                "all_time_total_requests": 0,
                "by_model": {},
                "by_request_type": {},
                "session_id_filter": session_id
            }
            return JSONResponse(content=frontend_summary)

        # Process log records
        logger.info(f"Processing {len(log_records)} log records")
        for row_num, row in enumerate(log_records):
            try:
                # Basic validation of required fields
                missing_fields = []
                for field in ["Token Cost USD", "Fixed Cost USD", "Input Tokens", "Output Tokens", 
                             "Grounded Searches Performed", "Model Used", "Request Type"]:
                    if field not in row:
                        missing_fields.append(field)
                
                if missing_fields:
                    summary["errors"].append(f"Row {row_num+2} missing fields: {missing_fields}")
                    continue
                
                summary["total_requests"] += 1
                
                # Use safer conversion with explicit error handling
                try:
                    token_cost = float(row.get("Token Cost USD", "0.0"))
                except (ValueError, TypeError):
                    summary["errors"].append(f"Row {row_num+2}: Invalid Token Cost: {row.get('Token Cost USD')}")
                    token_cost = 0.0
                
                try:
                    fixed_cost = float(row.get("Fixed Cost USD", "0.0"))
                except (ValueError, TypeError):
                    summary["errors"].append(f"Row {row_num+2}: Invalid Fixed Cost: {row.get('Fixed Cost USD')}")
                    fixed_cost = 0.0
                
                try:
                    input_tokens = int(row.get("Input Tokens", "0"))
                except (ValueError, TypeError):
                    summary["errors"].append(f"Row {row_num+2}: Invalid Input Tokens: {row.get('Input Tokens')}")
                    input_tokens = 0
                
                try:
                    output_tokens = int(row.get("Output Tokens", "0"))
                except (ValueError, TypeError):
                    summary["errors"].append(f"Row {row_num+2}: Invalid Output Tokens: {row.get('Output Tokens')}")
                    output_tokens = 0
                
                try:
                    grounded_searches = int(row.get("Grounded Searches Performed", "0"))
                except (ValueError, TypeError):
                    summary["errors"].append(f"Row {row_num+2}: Invalid Grounded Searches: {row.get('Grounded Searches Performed')}")
                    grounded_searches = 0
                
                model_used = row.get("Model Used", "unknown_model_in_log")
                request_type = row.get("Request Type", "unknown_type_in_log")
                timestamp_str = row.get("Timestamp", "")
                
                calculated_total_cost = token_cost + fixed_cost

                # Update summary totals
                summary["total_token_cost_usd"] += token_cost
                summary["total_fixed_cost_usd"] += fixed_cost
                summary["grand_total_estimated_cost_usd"] += calculated_total_cost
                summary["total_input_tokens"] += input_tokens
                summary["total_output_tokens"] += output_tokens
                summary["total_grounded_searches"] += grounded_searches

                # By Model
                summary["cost_by_model"][model_used]["requests"] += 1
                summary["cost_by_model"][model_used]["input_tokens"] += input_tokens
                summary["cost_by_model"][model_used]["output_tokens"] += output_tokens
                summary["cost_by_model"][model_used]["token_cost"] += token_cost
                summary["cost_by_model"][model_used]["fixed_cost"] += fixed_cost
                summary["cost_by_model"][model_used]["total_cost"] += calculated_total_cost
                summary["cost_by_model"][model_used]["grounded_searches"] += grounded_searches
                
                # By Request Type
                summary["cost_by_request_type"][request_type]["requests"] += 1
                summary["cost_by_request_type"][request_type]["input_tokens"] += input_tokens
                summary["cost_by_request_type"][request_type]["output_tokens"] += output_tokens
                summary["cost_by_request_type"][request_type]["token_cost"] += token_cost
                summary["cost_by_request_type"][request_type]["fixed_cost"] += fixed_cost
                summary["cost_by_request_type"][request_type]["total_cost"] += calculated_total_cost
                summary["cost_by_request_type"][request_type]["grounded_searches"] += grounded_searches

                # Process timestamp if available
                if timestamp_str:
                    try:
                        date_str = datetime.datetime.fromisoformat(timestamp_str).strftime('%Y-%m-%d')
                        summary["cost_by_date"][date_str]["requests"] += 1
                        summary["cost_by_date"][date_str]["token_cost"] += token_cost
                        summary["cost_by_date"][date_str]["fixed_cost"] += fixed_cost
                        summary["cost_by_date"][date_str]["total_cost"] += calculated_total_cost
                        summary["cost_by_date"][date_str]["grounded_searches"] += grounded_searches
                    except ValueError:
                        summary["errors"].append(f"Invalid timestamp format in row {row_num + 2}: {timestamp_str}")

            except Exception as row_error:
                # Catch any unexpected errors during row processing
                logger.error(f"Error processing row {row_num+2}: {row_error}", exc_info=True)
                summary["errors"].append(f"Error processing row {row_num+2}: {str(row_error)}")
        
        # Convert defaultdicts to regular dicts for serialization
        try:
            logger.debug("Converting defaultdicts to dicts")
            summary["cost_by_model"] = dict(summary["cost_by_model"])
            summary["cost_by_date"] = dict(summary["cost_by_date"])
            summary["cost_by_request_type"] = dict(summary["cost_by_request_type"])
        except Exception as convert_error:
            logger.error(f"Error converting defaultdicts: {convert_error}", exc_info=True)
            error_response["error"] = f"Internal conversion error: {str(convert_error)}"
            return JSONResponse(content=error_response, status_code=500)
        
        # Create the frontend-compatible summary format
        try:
            logger.debug("Creating frontend summary format")
            frontend_summary = {
                "all_time_total_token_cost": summary["total_token_cost_usd"],
                "all_time_total_fixed_cost": summary["total_fixed_cost_usd"],
                "all_time_total_cost": summary["grand_total_estimated_cost_usd"],
                "all_time_total_requests": summary["total_requests"],
                "by_model": {},
                "by_request_type": {},
                "session_id_filter": session_id
            }
            
            # Convert model data
            for model, data in summary["cost_by_model"].items():
                frontend_summary["by_model"][model] = {
                    "count": data["requests"],
                    "total_token_cost": data["token_cost"],
                    "total_fixed_cost": data["fixed_cost"],
                    "total_cost": data["total_cost"]
                }
            
            # Convert request type data
            for req_type, data in summary["cost_by_request_type"].items():
                frontend_summary["by_request_type"][req_type] = {
                    "count": data["requests"],
                    "total_token_cost": data["token_cost"],
                    "total_fixed_cost": data["fixed_cost"],
                    "total_cost": data["total_cost"]
                }
            
            # Include any errors if present
            if summary.get("errors", []):
                frontend_summary["errors"] = summary["errors"]
            
            logger.info("Successfully generated usage summary")
            return JSONResponse(content=frontend_summary)
            
        except Exception as format_error:
            logger.error(f"Error formatting frontend summary: {format_error}", exc_info=True)
            error_response["error"] = f"Error formatting summary: {str(format_error)}" 
            return JSONResponse(content=error_response, status_code=500)
            
    except Exception as e:
        # Top-level exception handler
        logger.error(f"Unhandled exception in usage summary: {e}", exc_info=True)
        error_response["error"] = f"Failed to generate usage summary: {str(e)}"
        return JSONResponse(content=error_response, status_code=500)

@app.get("/api/debug/gemini-config")
async def debug_gemini_config():
    """Debug endpoint to check Gemini API configuration"""
    import os
    import google.generativeai as genai
    
    result = {
        "google_api_key_present": bool(os.getenv("GOOGLE_API_KEY")),
        "google_api_key_length": len(os.getenv("GOOGLE_API_KEY", "")),
        "model_name": "gemini-2.5-flash"
    }
    
    # Test basic API configuration
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
            result["api_configuration_success"] = True
            
            # Test a very simple generation call
            try:
                test_response = model.generate_content("Say 'test'")
                result["simple_generation_success"] = True
                result["simple_generation_response"] = test_response.text if hasattr(test_response, 'text') else None
                result["has_usage_metadata"] = hasattr(test_response, 'usage_metadata')
                if hasattr(test_response, 'usage_metadata') and test_response.usage_metadata:
                    result["usage_metadata"] = {
                        "prompt_token_count": getattr(test_response.usage_metadata, 'prompt_token_count', None),
                        "candidates_token_count": getattr(test_response.usage_metadata, 'candidates_token_count', None)
                    }
            except Exception as e:
                result["simple_generation_success"] = False
                result["simple_generation_error"] = str(e)
        else:
            result["api_configuration_success"] = False
            result["error"] = "No API key found"
    except Exception as e:
        result["api_configuration_success"] = False
        result["error"] = str(e)
    
    return result

@app.get("/api/debug/pdf-cache")
async def debug_pdf_cache(clear: bool = Query(False)):
    """Debug endpoint to view and optionally clear PDF link extraction cache"""
    try:
        from PDF_link_extraction import get_cache_stats, clear_cache
    except ImportError:
        from .PDF_link_extraction import get_cache_stats, clear_cache
    
    # Get cache statistics before any potential clearing
    stats = get_cache_stats()
    
    response = {
        "cache_stats": stats,
        "message": "Cache statistics retrieved"
    }
    
    # Clear cache if requested
    if clear:
        clear_cache()
        response["message"] = "Cache cleared and statistics retrieved"
        response["action"] = "cache_cleared"
    
    return response

@app.post("/api/download-individual-pdf")
async def download_individual_pdf(
    request_data: DownloadIndividualPDFRequest,
    request: Request,
    current_user: User = Depends(get_current_active_user)
):
    """Downloads an individual PDF from base64 data."""
    try:
        logger.info(f"=== Individual PDF Download Request ===")
        logger.info(f"Filename: {request_data.filename}")
        logger.info(f"PDF data length: {len(request_data.pdf_data) if request_data.pdf_data else 0} characters")

        # Additional validation checks
        if not request_data.pdf_data:
            logger.error("PDF data parameter is empty or None")
            raise HTTPException(status_code=400, detail="PDF data is required")

        if not request_data.filename:
            logger.error("Filename parameter is empty or None")
            raise HTTPException(status_code=400, detail="Filename is required")

        logger.info(f"PDF data starts with: '{request_data.pdf_data[:20]}...' (first 20 characters)")

        # More permissive base64 validation - just check if it looks like base64
        if not re.match(r'^[A-Za-z0-9+/=]+$', request_data.pdf_data):
            logger.error(f"PDF data contains invalid base64 characters. Sample: {request_data.pdf_data[:100]}")
            raise HTTPException(status_code=400, detail="PDF data must be valid base64 encoded")

        logger.info(f"Attempting to decode base64 data of length {len(request_data.pdf_data)}")

        # Decode base64 data
        try:
            pdf_bytes = base64.b64decode(request_data.pdf_data, validate=True)
            logger.info(f"Successfully decoded base64 data. Resulting byte length: {len(pdf_bytes)}")
        except Exception as e:
            logger.error(f"Failed to decode base64 PDF data: {e}")
            logger.error(f"PDF data sample (first 100 chars): {request_data.pdf_data[:100]}")
            raise HTTPException(status_code=400, detail=f"Invalid base64 PDF data: {str(e)}")

        if not pdf_bytes:
            logger.error("Decoded PDF data is empty")
            raise HTTPException(status_code=400, detail="PDF data is empty after decoding")

        logger.info(f"Decoded data starts with: {pdf_bytes[:20]} (first 20 bytes)")

        # Validate that this looks like a PDF (more lenient check)
        if not pdf_bytes.startswith(b'%PDF'):
            logger.warning(f"Decoded data doesn't start with PDF header. First 20 bytes: {pdf_bytes[:20]}")
            pdf_start = pdf_bytes.find(b'%PDF')
            if pdf_start > 0 and pdf_start < 1000:
                logger.info(f"Found PDF header at position {pdf_start}, trimming leading bytes")
                pdf_bytes = pdf_bytes[pdf_start:]
            else:
                logger.warning("No PDF header found in first 1000 bytes, proceeding anyway")

        # Ensure filename has .pdf extension
        filename = request_data.filename
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'

        safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()
        if not safe_filename:
            safe_filename = "download.pdf"

        logger.info(f"Returning PDF download: {safe_filename} ({len(pdf_bytes)} bytes)")

        # AUDIT LOG (success)
        await log_api_success(
            request=request,
            action="smartSearchSingleDownload",
            entity_type="PDF",
            entity_id=safe_filename,
            metadata={
                "filename": safe_filename,
                "pdf_data_length": len(request_data.pdf_data),
                "decoded_pdf_bytes": len(pdf_bytes)
            },
            cost=0.0  # Set to actual cost if you have a pricing model
        )

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=\"{safe_filename}\"",
                "Content-Length": str(len(pdf_bytes))
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during individual PDF download: {e}", exc_info=True)
        logger.error(f"Request details - filename: {getattr(request_data, 'filename', 'N/A')}, "
                     f"pdf_data_length: {len(getattr(request_data, 'pdf_data', '') or '')}")
        raise HTTPException(status_code=500, detail=f"An error occurred during PDF download: {str(e)}")

@app.post("/api/test-download-individual-pdf")
async def test_download_individual_pdf():
    """Test endpoint for download-individual-pdf functionality with minimal test data."""
    try:
        # Create a simple test PDF (minimal valid PDF content)
        test_pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Test PDF) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f 
0000000015 00000 n 
0000000068 00000 n 
0000000125 00000 n 
0000000213 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
304
%%EOF"""
        
        # Encode to base64
        import base64
        test_pdf_base64 = base64.b64encode(test_pdf_content).decode('utf-8')
        
        logger.info(f"Test PDF created: {len(test_pdf_content)} bytes, base64 length: {len(test_pdf_base64)}")
        
        # Create FormData-like structure for internal testing
        from fastapi import Form
        
        # Call the actual endpoint function
        request = DownloadIndividualPDFRequest(
            pdf_data=test_pdf_base64,
            filename="test-download.pdf"
        )
        response = await download_individual_pdf(request)
        
        return {
            "success": True,
            "message": "Test PDF download endpoint is working",
            "test_pdf_size": len(test_pdf_content),
            "test_base64_length": len(test_pdf_base64),
            "response_type": str(type(response).__name__)
        }
        
    except Exception as e:
        logger.error(f"Test download endpoint failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "message": "Test PDF download endpoint failed"
        }

@app.post("/api/generate-validation-report")
async def generate_validation_report(
    request: GenerateValidationReportRequest,
    client: httpx.AsyncClient = Depends(lambda: app.state.httpx_client)
):
    """Generate a validation report PDF with the original PDF attached."""
    try:
        logger.info(f"Generating validation report for product: {request.product_name}")
        logger.info(f"Original PDF URL: {request.original_pdf_url}")
        
        # Download the original PDF
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = await client.get(request.original_pdf_url, headers=headers, timeout=300.0, follow_redirects=True)  # Increased to 5 minutes
            response.raise_for_status()
            original_pdf_bytes = response.content
        except Exception as e:
            logger.error(f"Failed to download original PDF: {e}")
            raise HTTPException(status_code=502, detail=f"Failed to download original PDF: {str(e)}")
        
        if len(original_pdf_bytes) == 0:
            raise HTTPException(status_code=502, detail="Downloaded PDF is empty")
        
        # Generate the validation report and merge with original PDF
        try:
            merged_pdf_bytes = await run_in_threadpool(
                generate_validation_report_and_merge,
                validation_data=request.validation_data,
                product_name=request.product_name,
                original_pdf_bytes=original_pdf_bytes
            )
            
            if not merged_pdf_bytes:
                raise HTTPException(status_code=500, detail="Failed to generate validation report")
            
            logger.info(f"Generated validation report PDF: {len(merged_pdf_bytes)} bytes")
            
            # Return the merged PDF as a downloadable response
            return StreamingResponse(
                io.BytesIO(merged_pdf_bytes),
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f"attachment; filename=\"{request.product_name}_validation_report.pdf\"",
                    "Content-Length": str(len(merged_pdf_bytes))
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating validation report: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error generating validation report: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_validation_report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.post("/api/generate-smart-validation-report")
async def generate_smart_validation_report(
    request: GenerateSmartValidationReportRequest,
    client: httpx.AsyncClient = Depends(lambda: app.state.httpx_client)
):
    """Generate a validation report PDF with the original PDF attached."""
    start_time = time.time()
    try:
        logger.info(f"Generating validation report for product: {request.product_name}")

        
        # Decode the base64 PDF bytes
        try:
            original_pdf_bytes = base64.b64decode(request.original_pdf_bytes)
            if len(original_pdf_bytes) == 0:
                raise HTTPException(status_code=400, detail="PDF bytes are empty")
        except Exception as e:
            logger.error(f"Failed to decode PDF bytes: {e}")
            error_msg = f"Failed to decode PDF bytes: {e}"
            logger.error(error_msg)
            # await log_api_failure(
            #     request=request,
            #     action="SmartValidation",
            #     entity_type="PDF",
            #     entity_id=report_request.product_name,
            #     error=error_msg,
            #     start_time=start_time
            # )
            raise HTTPException(status_code=400, detail=f"Invalid PDF data: {str(e)}")

        # Generate the validation report and merge with original PDF
        try:
            merged_pdf_bytes = await run_in_threadpool(
                generate_validation_report_and_merge,
                validation_data=request.validation_data,
                product_name=request.product_name,
                original_pdf_bytes=original_pdf_bytes
            )
            
            if not merged_pdf_bytes:
                raise HTTPException(status_code=500, detail="Failed to generate validation report")
            
            # Calculate processing metrics
            process_time = int((time.time() - start_time) * 1000)
            pdf_size = len(merged_pdf_bytes)
            # Log successful generation
            # await log_api_success(
            #     request=request,
            #     action="SmartValidation",
            #     entity_type="PDF",
            #     entity_id=report_request.product_name,
            #     metadata={
            #         "product": report_request.product_name,
            #         "pdf_size_bytes": pdf_size,
            #         "validation_items": len(report_request.validation_data),
            #         "generated_size_bytes": pdf_size
            #     },
            #     start_time=start_time
            # )
            
            logger.info(f"Generated validation report PDF: {len(merged_pdf_bytes)} bytes")
            
            # Return the merged PDF as a downloadable response
            return StreamingResponse(
                io.BytesIO(merged_pdf_bytes),
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f"attachment; filename=\"{request.product_name}_validation_report.pdf\"",
                    "Content-Length": str(len(merged_pdf_bytes))
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating validation report: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error generating validation report: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_validation_report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    

# --- Main Execution (for running with uvicorn directly) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server directly...")
    logger.info(f"Server will start on {settings.host}:{settings.port}")
    uvicorn.run("api_server:app", host=settings.host, port=settings.port, reload=settings.reload)

def excepthook(type, value, tb):
    print("UNCAUGHT EXCEPTION:", value)
    traceback.print_exception(type, value, tb)

sys.excepthook = excepthook