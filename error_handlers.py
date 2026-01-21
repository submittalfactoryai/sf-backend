"""
Standardized Error Handling for Submittal Factory
This module provides consistent error codes and messages for the frontend.

Author: Submittal Factory Team
Version: 1.0.0
"""

from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel
from fastapi import HTTPException
from fastapi.responses import JSONResponse
import logging
import datetime
import uuid

logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    """Categories of errors for classification and UI grouping."""
    UPLOAD = "UPLOAD"
    EXTRACTION = "EXTRACTION"
    LLM = "LLM"
    SEARCH = "SEARCH"
    VALIDATION = "VALIDATION"
    NETWORK = "NETWORK"
    SYSTEM = "SYSTEM"


class ErrorCode(str, Enum):
    """
    Standardized error codes.
    
    Code ranges:
    - 1xxx: Upload errors
    - 2xxx: Extraction errors
    - 3xxx: LLM/AI errors
    - 4xxx: Search errors
    - 5xxx: Validation errors
    - 6xxx: Network errors
    - 9xxx: System errors
    """
    
    # Upload Errors (1xxx)
    FILE_TOO_LARGE = "1001"
    TOO_MANY_PRODUCTS = "1002"
    INVALID_FILE_FORMAT = "1003"
    EMPTY_FILE = "1004"
    CORRUPTED_FILE = "1005"
    
    # Extraction Errors (2xxx)
    PDF_NOT_READABLE = "2001"
    SCANNED_DOCUMENT = "2002"
    NO_TEXT_CONTENT = "2003"
    NO_PART2_FOUND = "2004"
    EXTRACTION_FAILED = "2005"
    
    # LLM/AI Errors (3xxx)
    MODEL_NOT_FOUND = "3001"
    MODEL_OVERLOADED = "3002"
    API_KEY_INVALID = "3003"
    RATE_LIMITED = "3004"
    CONTEXT_TOO_LONG = "3005"
    LLM_RESPONSE_INVALID = "3006"
    LLM_TIMEOUT = "3007"
    LLM_SAFETY_BLOCKED = "3008"
    
    # Search Errors (4xxx)
    NO_RESULTS_FOUND = "4001"
    SEARCH_TIMEOUT = "4002"
    INVALID_SEARCH_PARAMS = "4003"
    
    # Validation Errors (5xxx)
    VALIDATION_FAILED = "5001"
    SPECS_MISMATCH = "5002"
    PDF_FETCH_FAILED = "5003"
    INCOMPLETE_PRODUCT_DATA = "5004"
    
    # Network Errors (6xxx)
    CONNECTION_FAILED = "6001"
    TIMEOUT = "6002"
    DNS_FAILURE = "6003"
    SSL_ERROR = "6004"
    PROXY_ERROR = "6005"
    
    # System Errors (9xxx)
    INTERNAL_ERROR = "9001"
    DATABASE_ERROR = "9002"
    CONFIGURATION_ERROR = "9003"
    SUBSCRIPTION_ERROR = "9004"


# User-friendly messages mapped to error codes (SOW Section 6 compliant)
ERROR_MESSAGES: Dict[str, Dict[str, str]] = {
    # ==========================================
    # Upload Limits (SOW 6a)
    # ==========================================
    ErrorCode.FILE_TOO_LARGE: {
        "user_message": "File size or length exceeds system limits. Please reduce document size or split into smaller files.",
        "severity": "error",
        "action": "Please upload a file smaller than 5MB.",
        "category": "UPLOAD"
    },
    ErrorCode.TOO_MANY_PRODUCTS: {
        "user_message": "The uploaded file contains more than 100 products. Please upload a smaller file to ensure accurate extraction.",
        "severity": "warning",
        "action": "Consider splitting your specification into multiple documents.",
        "category": "UPLOAD"
    },
    ErrorCode.INVALID_FILE_FORMAT: {
        "user_message": "Extraction failed due to unsupported file format or corrupted content. Please upload a valid PDF.",
        "severity": "error",
        "action": "Only PDF files are supported.",
        "category": "UPLOAD"
    },
    ErrorCode.EMPTY_FILE: {
        "user_message": "The uploaded file appears to be empty or contains no readable content.",
        "severity": "error",
        "action": "Please check your file and upload again.",
        "category": "UPLOAD"
    },
    ErrorCode.CORRUPTED_FILE: {
        "user_message": "The uploaded file appears to be corrupted and cannot be processed.",
        "severity": "error",
        "action": "Please try uploading a different copy of the file.",
        "category": "UPLOAD"
    },
    
    # ==========================================
    # Extraction Boundaries (SOW 6b)
    # ==========================================
    ErrorCode.PDF_NOT_READABLE: {
        "user_message": "The system's extraction engine supports machine-readable, text-based PDFs only.",
        "severity": "error",
        "action": "Please ensure your PDF contains selectable text, not just images.",
        "category": "EXTRACTION"
    },
    ErrorCode.SCANNED_DOCUMENT: {
        "user_message": "Scanned images, handwritten notes, markups, or heavily annotated documents are unsupported and may yield inaccurate results.",
        "severity": "warning",
        "action": "For best results, use text-based PDFs created from digital sources.",
        "category": "EXTRACTION"
    },
    ErrorCode.NO_TEXT_CONTENT: {
        "user_message": "No readable text content was found in the uploaded document.",
        "severity": "error",
        "action": "Please ensure the PDF contains machine-readable text.",
        "category": "EXTRACTION"
    },
    ErrorCode.NO_PART2_FOUND: {
        "user_message": "Could not locate 'PART 2 - PRODUCTS' section in the document.",
        "severity": "warning",
        "action": "Ensure your specification follows standard CSI format with a PART 2 section.",
        "category": "EXTRACTION"
    },
    ErrorCode.EXTRACTION_FAILED: {
        "user_message": "Extraction failed due to document structure issues.",
        "severity": "error",
        "action": "Please verify the document format and try again.",
        "category": "EXTRACTION"
    },
    
    # ==========================================
    # LLM Product Extraction (SOW 6c)
    # ==========================================
    ErrorCode.MODEL_NOT_FOUND: {
        "user_message": "The AI extraction service is temporarily unavailable. Our team has been notified.",
        "severity": "error",
        "action": "Please try again in a few minutes. If the issue persists, contact support.",
        "category": "LLM",
        "technical_hint": "Model configuration issue - check GOOGLE_API_KEY and model name"
    },
    ErrorCode.MODEL_OVERLOADED: {
        "user_message": "The AI service is experiencing high demand. Please wait a moment and try again.",
        "severity": "warning",
        "action": "Retry in 30 seconds.",
        "category": "LLM"
    },
    ErrorCode.API_KEY_INVALID: {
        "user_message": "AI service authentication failed. Please contact support.",
        "severity": "error",
        "action": "Contact your administrator.",
        "category": "LLM"
    },
    ErrorCode.RATE_LIMITED: {
        "user_message": "You've reached the temporary request limit. Please wait before trying again.",
        "severity": "warning",
        "action": "Wait 1-2 minutes before retrying.",
        "category": "LLM"
    },
    ErrorCode.CONTEXT_TOO_LONG: {
        "user_message": "The document is too large for AI processing. Please upload a shorter specification.",
        "severity": "error",
        "action": "Split your document into smaller sections.",
        "category": "LLM"
    },
    ErrorCode.LLM_RESPONSE_INVALID: {
        "user_message": "Unable to extract product details from the uploaded document. Please ensure product specifications are clearly formatted in the PDF or Product Data Sheet.",
        "severity": "error",
        "action": "Check that your specification has clear product listings.",
        "category": "LLM"
    },
    ErrorCode.LLM_TIMEOUT: {
        "user_message": "The AI processing took too long. Please try with a smaller document.",
        "severity": "error",
        "action": "Try uploading a smaller file or splitting into sections.",
        "category": "LLM"
    },
    ErrorCode.LLM_SAFETY_BLOCKED: {
        "user_message": "The AI system flagged content in the document. Please review and try again.",
        "severity": "warning",
        "action": "Contact support if you believe this is an error.",
        "category": "LLM"
    },
    
    # ==========================================
    # Search Results (SOW 6d)
    # ==========================================
    ErrorCode.NO_RESULTS_FOUND: {
        "user_message": "No matching submittal documents found for the selected product and specifications.",
        "severity": "info",
        "action": "Try broadening your search criteria or check the product specifications.",
        "category": "SEARCH"
    },
    ErrorCode.SEARCH_TIMEOUT: {
        "user_message": "Search took too long to complete. Please try again.",
        "severity": "warning",
        "action": "Retry the search or try with fewer specifications.",
        "category": "SEARCH"
    },
    ErrorCode.INVALID_SEARCH_PARAMS: {
        "user_message": "Product specifications or manufacturer details are incomplete or missing in the file.",
        "severity": "error",
        "action": "Ensure the product has valid specifications before searching.",
        "category": "SEARCH"
    },
    
    # ==========================================
    # Validation Errors (SOW 6e)
    # ==========================================
    ErrorCode.VALIDATION_FAILED: {
        "user_message": "Validation failed: The uploaded document does not meet technical specification criteria for the selected product.",
        "severity": "error",
        "action": "Review the specifications and try a different submittal document.",
        "category": "VALIDATION"
    },
    ErrorCode.SPECS_MISMATCH: {
        "user_message": "Partial validation completed. Some inconsistencies were detected in manufacturer or product details.",
        "severity": "warning",
        "action": "Review the matched and unmatched specifications in the results.",
        "category": "VALIDATION"
    },
    ErrorCode.PDF_FETCH_FAILED: {
        "user_message": "Could not retrieve the PDF document for validation.",
        "severity": "error",
        "action": "Check if the PDF link is still valid and accessible.",
        "category": "VALIDATION"
    },
    ErrorCode.INCOMPLETE_PRODUCT_DATA: {
        "user_message": "Product specifications or manufacturer details are incomplete or missing.",
        "severity": "warning",
        "action": "Ensure complete product information is available for validation.",
        "category": "VALIDATION"
    },
    
    # ==========================================
    # Network/Timeout (SOW 6f)
    # ==========================================
    ErrorCode.CONNECTION_FAILED: {
        "user_message": "Network or backend timeout occurred. Please try again later.",
        "severity": "error",
        "action": "Check your internet connection and retry.",
        "category": "NETWORK"
    },
    ErrorCode.TIMEOUT: {
        "user_message": "The request timed out. Please try again.",
        "severity": "error",
        "action": "The server is taking longer than expected. Please retry.",
        "category": "NETWORK"
    },
    ErrorCode.DNS_FAILURE: {
        "user_message": "Could not connect to the server. Please check your network.",
        "severity": "error",
        "action": "Verify your internet connection.",
        "category": "NETWORK"
    },
    ErrorCode.SSL_ERROR: {
        "user_message": "Secure connection failed. Please contact support if this persists.",
        "severity": "error",
        "action": "Try refreshing the page or contact support.",
        "category": "NETWORK"
    },
    ErrorCode.PROXY_ERROR: {
        "user_message": "Connection through proxy failed.",
        "severity": "error",
        "action": "Check your network configuration.",
        "category": "NETWORK"
    },
    
    # ==========================================
    # System Errors
    # ==========================================
    ErrorCode.INTERNAL_ERROR: {
        "user_message": "The system experienced an unexpected error. Contact support with the error details if the problem persists.",
        "severity": "error",
        "action": "Please try again. If the issue continues, contact support with error reference.",
        "category": "SYSTEM"
    },
    ErrorCode.DATABASE_ERROR: {
        "user_message": "A database error occurred. Please try again.",
        "severity": "error",
        "action": "Retry your request.",
        "category": "SYSTEM"
    },
    ErrorCode.CONFIGURATION_ERROR: {
        "user_message": "System configuration issue detected. Our team has been notified.",
        "severity": "error",
        "action": "Please try again later or contact support.",
        "category": "SYSTEM"
    },
    ErrorCode.SUBSCRIPTION_ERROR: {
        "user_message": "Subscription verification failed. Please contact your administrator.",
        "severity": "error",
        "action": "Check your subscription status or contact support.",
        "category": "SYSTEM"
    },
}


class StandardErrorResponse(BaseModel):
    """Standardized error response format for frontend consumption."""
    success: bool = False
    error_code: str
    error_category: str
    user_message: str
    action_required: Optional[str] = None
    severity: str = "error"  # error, warning, info
    technical_details: Optional[str] = None  # Only in debug mode
    reference_id: Optional[str] = None  # For support tracking
    timestamp: Optional[str] = None


def create_error_response(
    error_code: ErrorCode,
    technical_details: Optional[str] = None,
    additional_context: Optional[Dict[str, Any]] = None,
    include_technical: bool = False
) -> Dict[str, Any]:
    """
    Create a standardized error response dictionary.
    
    Args:
        error_code: The specific error code
        technical_details: Technical error message (for logging)
        additional_context: Any additional data to include
        include_technical: Whether to include technical details in response
    
    Returns:
        Dictionary formatted for JSONResponse
    """
    error_info = ERROR_MESSAGES.get(error_code, {
        "user_message": "An unexpected error occurred.",
        "severity": "error",
        "action": "Please try again or contact support.",
        "category": "SYSTEM"
    })
    
    # Determine category from error code or error info
    category = error_info.get("category", "SYSTEM")
    
    # Generate reference ID for tracking
    reference_id = str(uuid.uuid4())[:8].upper()
    
    response = {
        "success": False,
        "error_code": error_code.value if isinstance(error_code, ErrorCode) else str(error_code),
        "error_category": category,
        "user_message": error_info["user_message"],
        "action_required": error_info.get("action"),
        "severity": error_info.get("severity", "error"),
        "reference_id": reference_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    
    if include_technical and technical_details:
        response["technical_details"] = technical_details
    
    if additional_context:
        # Don't override core fields
        safe_context = {k: v for k, v in additional_context.items() 
                       if k not in ["success", "error_code", "error_category", "user_message"]}
        response.update(safe_context)
    
    # Log the error with full technical details
    logger.error(
        f"Error Response Generated - Code: {error_code}, "
        f"Ref: {reference_id}, "
        f"Category: {category}, "
        f"Technical: {technical_details}"
    )
    
    return response


def classify_llm_error(error: Exception) -> ErrorCode:
    """
    Classify LLM/Gemini errors into appropriate error codes.
    
    This is crucial for converting technical errors like "404 model not found"
    into user-friendly error codes.
    
    Args:
        error: The exception from the LLM call
        
    Returns:
        ErrorCode: The appropriate error code for the error
    """
    error_str = str(error).lower()
    error_type = type(error).__name__
    
    logger.debug(f"Classifying LLM error: {error_type} - {error_str[:200]}")
    
    # Model not found / 404 errors - CRITICAL FOR YOUR ISSUE
    if "404" in error_str:
        if "model" in error_str or "resource" in error_str:
            logger.warning(f"Model not found error detected: {error}")
            return ErrorCode.MODEL_NOT_FOUND
    
    if "not found" in error_str and "model" in error_str:
        logger.warning(f"Model not found error detected: {error}")
        return ErrorCode.MODEL_NOT_FOUND
    
    # Check for specific Gemini model errors
    if "gemini" in error_str and ("unavailable" in error_str or "not available" in error_str):
        return ErrorCode.MODEL_NOT_FOUND
    
    # Authentication errors
    if any(x in error_str for x in ["401", "403", "unauthorized", "api key", "authentication", "permission"]):
        return ErrorCode.API_KEY_INVALID
    
    # Rate limiting
    if any(x in error_str for x in ["429", "rate limit", "quota", "too many requests", "resource exhausted"]):
        return ErrorCode.RATE_LIMITED
    
    # Timeout errors
    if any(x in error_str for x in ["timeout", "deadline", "timed out", "deadline exceeded"]):
        return ErrorCode.LLM_TIMEOUT
    
    # Context/token limit
    if ("token" in error_str or "context" in error_str) and ("limit" in error_str or "exceed" in error_str or "length" in error_str):
        return ErrorCode.CONTEXT_TOO_LONG
    
    # Model overloaded
    if any(x in error_str for x in ["503", "overloaded", "capacity", "service unavailable", "temporarily unavailable"]):
        return ErrorCode.MODEL_OVERLOADED
    
    # Safety/content filtered
    if any(x in error_str for x in ["safety", "blocked", "filtered", "harm_category"]):
        return ErrorCode.LLM_SAFETY_BLOCKED
    
    # Invalid response / JSON parsing
    if any(x in error_str for x in ["json", "parse", "invalid response", "malformed", "decode"]):
        return ErrorCode.LLM_RESPONSE_INVALID
    
    # Connection errors
    if any(x in error_str for x in ["connection", "network", "unreachable"]):
        return ErrorCode.CONNECTION_FAILED
    
    # Default to internal error
    logger.warning(f"Unclassified LLM error, defaulting to INTERNAL_ERROR: {error}")
    return ErrorCode.INTERNAL_ERROR


def classify_network_error(error: Exception) -> ErrorCode:
    """
    Classify network-related errors.
    
    Args:
        error: The network exception
        
    Returns:
        ErrorCode: The appropriate error code
    """
    error_str = str(error).lower()
    
    if any(x in error_str for x in ["timeout", "timed out", "deadline"]):
        return ErrorCode.TIMEOUT
    
    if any(x in error_str for x in ["connection refused", "connection reset", "connection error"]):
        return ErrorCode.CONNECTION_FAILED
    
    if any(x in error_str for x in ["dns", "resolve", "getaddrinfo", "name resolution"]):
        return ErrorCode.DNS_FAILURE
    
    if any(x in error_str for x in ["ssl", "certificate", "tls", "handshake"]):
        return ErrorCode.SSL_ERROR
    
    if "proxy" in error_str:
        return ErrorCode.PROXY_ERROR
    
    return ErrorCode.CONNECTION_FAILED


def classify_extraction_error(error: Exception, context: Optional[str] = None) -> ErrorCode:
    """
    Classify PDF extraction errors.
    
    Args:
        error: The extraction exception
        context: Optional context about what was being extracted
        
    Returns:
        ErrorCode: The appropriate error code
    """
    error_str = str(error).lower()
    
    if any(x in error_str for x in ["scanned", "image", "ocr needed", "no text"]):
        return ErrorCode.SCANNED_DOCUMENT
    
    if any(x in error_str for x in ["corrupt", "damaged", "invalid pdf", "malformed"]):
        return ErrorCode.CORRUPTED_FILE
    
    if any(x in error_str for x in ["empty", "no content", "blank"]):
        return ErrorCode.NO_TEXT_CONTENT
    
    if any(x in error_str for x in ["part 2", "products section", "section not found"]):
        return ErrorCode.NO_PART2_FOUND
    
    if any(x in error_str for x in ["cannot read", "unreadable", "encrypted", "password"]):
        return ErrorCode.PDF_NOT_READABLE
    
    return ErrorCode.EXTRACTION_FAILED


class SubmittalFactoryException(Exception):
    """
    Custom exception class for Submittal Factory errors.
    
    This provides a consistent way to raise and handle errors throughout the application.
    
    Usage:
        raise SubmittalFactoryException(
            ErrorCode.MODEL_NOT_FOUND,
            "Gemini model gemini-2.5-flash not available",
            {"model_name": "gemini-2.5-flash"}
        )
    """
    
    def __init__(
        self,
        error_code: ErrorCode,
        technical_message: str = "",
        additional_data: Optional[Dict[str, Any]] = None
    ):
        self.error_code = error_code
        self.technical_message = technical_message
        self.additional_data = additional_data or {}
        
        # Get user message for the exception message
        error_info = ERROR_MESSAGES.get(error_code, {})
        user_message = error_info.get("user_message", "An unexpected error occurred.")
        
        super().__init__(f"{user_message} (Technical: {technical_message})")
    
    def to_response(self, include_technical: bool = False) -> JSONResponse:
        """
        Convert to FastAPI JSONResponse.
        
        Args:
            include_technical: Whether to include technical details in response
            
        Returns:
            JSONResponse with appropriate status code and error data
        """
        error_info = ERROR_MESSAGES.get(self.error_code, {})
        
        # Map error codes to HTTP status codes
        status_code_map = {
            "1": 400,  # Upload errors - Bad Request
            "2": 422,  # Extraction errors - Unprocessable Entity
            "3": 503,  # LLM errors - Service Unavailable
            "4": 404,  # Search errors - Not Found (can also be 200 with empty results)
            "5": 422,  # Validation errors - Unprocessable Entity
            "6": 502,  # Network errors - Bad Gateway
            "9": 500,  # System errors - Internal Server Error
        }
        
        code_value = self.error_code.value if isinstance(self.error_code, ErrorCode) else str(self.error_code)
        code_prefix = code_value[0]
        status_code = status_code_map.get(code_prefix, 500)
        
        # Special case: NO_RESULTS_FOUND should be 200 with info severity
        if self.error_code == ErrorCode.NO_RESULTS_FOUND:
            status_code = 200
        
        response_data = create_error_response(
            self.error_code,
            self.technical_message,
            self.additional_data,
            include_technical
        )
        
        return JSONResponse(
            status_code=status_code,
            content=response_data
        )
    
    def to_dict(self, include_technical: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary format.
        
        Args:
            include_technical: Whether to include technical details
            
        Returns:
            Dictionary with error information
        """
        return create_error_response(
            self.error_code,
            self.technical_message,
            self.additional_data,
            include_technical
        )


def handle_gemini_exception(error: Exception, context: str = "") -> SubmittalFactoryException:
    """
    Convert a Gemini API exception to a SubmittalFactoryException.
    
    This is the main function to use when catching Gemini errors.
    
    Args:
        error: The original exception
        context: Additional context about what operation was being performed
        
    Returns:
        SubmittalFactoryException with appropriate error code
    """
    error_code = classify_llm_error(error)
    
    technical_message = f"{context}: {str(error)}" if context else str(error)
    
    return SubmittalFactoryException(
        error_code=error_code,
        technical_message=technical_message,
        additional_data={
            "original_error_type": type(error).__name__,
            "context": context
        }
    )


# Convenience functions for common error scenarios
def upload_error(code: ErrorCode, filename: str = "", details: str = "") -> SubmittalFactoryException:
    """Create an upload-related error."""
    return SubmittalFactoryException(
        error_code=code,
        technical_message=f"File: {filename}. {details}",
        additional_data={"filename": filename}
    )


def extraction_error(code: ErrorCode, filename: str = "", details: str = "") -> SubmittalFactoryException:
    """Create an extraction-related error."""
    return SubmittalFactoryException(
        error_code=code,
        technical_message=f"Extraction from {filename} failed: {details}",
        additional_data={"filename": filename}
    )


def validation_error(code: ErrorCode, product_name: str = "", details: str = "") -> SubmittalFactoryException:
    """Create a validation-related error."""
    return SubmittalFactoryException(
        error_code=code,
        technical_message=f"Validation for '{product_name}' failed: {details}",
        additional_data={"product_name": product_name}
    )


def search_error(code: ErrorCode, query: str = "", details: str = "") -> SubmittalFactoryException:
    """Create a search-related error."""
    return SubmittalFactoryException(
        error_code=code,
        technical_message=f"Search failed: {details}",
        additional_data={"query": query}
    )