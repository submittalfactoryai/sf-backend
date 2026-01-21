"""
Extraction Warnings and Response Helpers
=========================================

This module provides warning messages, response helpers, and error handling
for the extraction pipeline. It ensures consistent, professional messaging
to users about extraction results, warnings, and errors.

Author: Submittal Factory Team
Date: 2025
"""

import logging
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# WARNING TYPES AND THRESHOLDS
# =============================================================================

class WarningType(Enum):
    """Enum for different types of warnings."""
    LARGE_PRODUCT_COUNT = "large_product_count"
    NON_STANDARD_SECTION = "non_standard_section"
    LOW_CONFIDENCE_DETECTION = "low_confidence_detection"
    FALLBACK_DETECTION = "fallback_detection"
    TRUNCATED_CONTENT = "truncated_content"
    PARTIAL_EXTRACTION = "partial_extraction"
    MISSING_DATA = "missing_data"
    FILE_TOO_LARGE = "file_too_large"
    PROCESSING_TIMEOUT = "processing_timeout"
    RETRY_REQUIRED = "retry_required"


class WarningSeverity(Enum):
    """Severity levels for warnings."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# Thresholds
PRODUCT_COUNT_WARNING_THRESHOLD = 100
PRODUCT_COUNT_CRITICAL_THRESHOLD = 200
LOW_CONFIDENCE_THRESHOLD = 0.6
FILE_SIZE_WARNING_MB = 10
FILE_SIZE_CRITICAL_MB = 25


# =============================================================================
# WARNING DATA CLASSES
# =============================================================================

@dataclass
class ExtractionWarning:
    """Represents a single warning from the extraction process."""
    type: WarningType
    severity: WarningSeverity
    message: str
    details: Optional[str] = None
    user_message: str = ""  # User-friendly message
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "user_message": self.user_message
        }


@dataclass
class ExtractionResult:
    """Represents the complete result of an extraction operation."""
    success: bool
    products: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[ExtractionWarning] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_warning(self, warning: ExtractionWarning):
        """Add a warning to the result."""
        self.warnings.append(warning)
        
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0
    
    def has_critical_warnings(self) -> bool:
        """Check if there are any critical warnings."""
        return any(w.severity == WarningSeverity.CRITICAL for w in self.warnings)
    
    def get_warnings_by_severity(self, severity: WarningSeverity) -> List[ExtractionWarning]:
        """Get warnings filtered by severity."""
        return [w for w in self.warnings if w.severity == severity]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "products_count": len(self.products),
            "warnings": [w.to_dict() for w in self.warnings],
            "errors": self.errors,
            "has_warnings": self.has_warnings(),
            "has_critical_warnings": self.has_critical_warnings()
        }


# =============================================================================
# WARNING GENERATORS
# =============================================================================

def generate_large_product_count_warning(product_count: int) -> Optional[ExtractionWarning]:
    """
    Generate a warning if the product count exceeds thresholds.
    
    Args:
        product_count: Number of products extracted
        
    Returns:
        ExtractionWarning or None if no warning needed
    """
    if product_count >= PRODUCT_COUNT_CRITICAL_THRESHOLD:
        return ExtractionWarning(
            type=WarningType.LARGE_PRODUCT_COUNT,
            severity=WarningSeverity.CRITICAL,
            message=f"Extracted {product_count} products, which exceeds the critical threshold of {PRODUCT_COUNT_CRITICAL_THRESHOLD}",
            details=f"Product count: {product_count}",
            user_message=(
                f"⚠️ Large Document Warning: This extraction returned {product_count} products. "
                "Due to the document's size, some product details may be incomplete or missing. "
                "For best results, consider splitting the document into smaller sections or "
                "reviewing the extracted data carefully for completeness."
            )
        )
    elif product_count >= PRODUCT_COUNT_WARNING_THRESHOLD:
        return ExtractionWarning(
            type=WarningType.LARGE_PRODUCT_COUNT,
            severity=WarningSeverity.WARNING,
            message=f"Extracted {product_count} products, which exceeds the warning threshold of {PRODUCT_COUNT_WARNING_THRESHOLD}",
            details=f"Product count: {product_count}",
            user_message=(
                f"📋 Note: This extraction returned {product_count} products. "
                "Large documents may occasionally have incomplete extractions. "
                "Please review the results to ensure all products are captured accurately."
            )
        )
    return None


def generate_non_standard_section_warning(
    detected_section: str,
    expected_section: str = "PART_2"
) -> ExtractionWarning:
    """
    Generate a warning when products are found in a non-standard section.
    
    Args:
        detected_section: The section where products were found
        expected_section: The expected section (usually PART 2)
        
    Returns:
        ExtractionWarning
    """
    section_display = detected_section.replace("_", " ")
    
    return ExtractionWarning(
        type=WarningType.NON_STANDARD_SECTION,
        severity=WarningSeverity.INFO,
        message=f"Products found in {detected_section} instead of {expected_section}",
        details=f"Detected section: {detected_section}",
        user_message=(
            f"ℹ️ This document has a non-standard structure. "
            f"Products were detected in {section_display} instead of the typical PART 2 - PRODUCTS section. "
            "The extraction has been adjusted accordingly."
        )
    )


def generate_low_confidence_warning(
    confidence: float,
    detection_method: str
) -> Optional[ExtractionWarning]:
    """
    Generate a warning for low confidence section detection.
    
    Args:
        confidence: Detection confidence score (0-1)
        detection_method: Method used for detection
        
    Returns:
        ExtractionWarning or None
    """
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return ExtractionWarning(
            type=WarningType.LOW_CONFIDENCE_DETECTION,
            severity=WarningSeverity.WARNING,
            message=f"Section detection confidence is low ({confidence:.0%})",
            details=f"Confidence: {confidence}, Method: {detection_method}",
            user_message=(
                f"⚠️ The document structure could not be determined with high confidence "
                f"(confidence level: {confidence:.0%}). "
                "Please verify that the extracted products match your expectations."
            )
        )
    return None


def generate_fallback_detection_warning(reason: str) -> ExtractionWarning:
    """
    Generate a warning when fallback detection was used.
    
    Args:
        reason: Reason for using fallback
        
    Returns:
        ExtractionWarning
    """
    return ExtractionWarning(
        type=WarningType.FALLBACK_DETECTION,
        severity=WarningSeverity.WARNING,
        message="Fallback detection method was used",
        details=reason,
        user_message=(
            "ℹ️ This document required special processing to detect the products section. "
            "Results have been extracted but may need additional review."
        )
    )


def generate_truncated_content_warning(
    original_length: int,
    truncated_length: int,
    max_length: int
) -> ExtractionWarning:
    """
    Generate a warning when content was truncated.
    
    Args:
        original_length: Original content length
        truncated_length: Length after truncation
        max_length: Maximum allowed length
        
    Returns:
        ExtractionWarning
    """
    return ExtractionWarning(
        type=WarningType.TRUNCATED_CONTENT,
        severity=WarningSeverity.WARNING,
        message=f"Content was truncated from {original_length} to {truncated_length} characters",
        details=f"Original: {original_length}, Truncated: {truncated_length}, Max: {max_length}",
        user_message=(
            f"⚠️ This document's content was too large to process completely "
            f"(original size exceeded {max_length:,} characters). "
            "Some products at the end of the document may not have been extracted. "
            "Consider processing this document in sections for complete extraction."
        )
    )


def generate_file_size_warning(file_size_mb: float) -> Optional[ExtractionWarning]:
    """
    Generate a warning based on file size.
    
    Args:
        file_size_mb: File size in megabytes
        
    Returns:
        ExtractionWarning or None
    """
    if file_size_mb >= FILE_SIZE_CRITICAL_MB:
        return ExtractionWarning(
            type=WarningType.FILE_TOO_LARGE,
            severity=WarningSeverity.CRITICAL,
            message=f"File size ({file_size_mb:.1f} MB) exceeds critical threshold",
            details=f"File size: {file_size_mb:.1f} MB",
            user_message=(
                f"⚠️ Large File Warning: This file ({file_size_mb:.1f} MB) is significantly larger "
                "than optimal for extraction. Processing may take longer and some content "
                "may not be fully extracted. For best results, consider splitting the document."
            )
        )
    elif file_size_mb >= FILE_SIZE_WARNING_MB:
        return ExtractionWarning(
            type=WarningType.FILE_TOO_LARGE,
            severity=WarningSeverity.WARNING,
            message=f"File size ({file_size_mb:.1f} MB) is large",
            details=f"File size: {file_size_mb:.1f} MB",
            user_message=(
                f"📋 Note: This is a large file ({file_size_mb:.1f} MB). "
                "Processing may take slightly longer than usual."
            )
        )
    return None


def generate_retry_warning(attempt: int, max_attempts: int) -> ExtractionWarning:
    """
    Generate a warning when retries were needed.
    
    Args:
        attempt: Current attempt number
        max_attempts: Maximum number of attempts
        
    Returns:
        ExtractionWarning
    """
    return ExtractionWarning(
        type=WarningType.RETRY_REQUIRED,
        severity=WarningSeverity.INFO,
        message=f"Extraction required {attempt} of {max_attempts} attempts",
        details=f"Attempt: {attempt}/{max_attempts}",
        user_message=""  # Silent warning, not shown to user
    )


# =============================================================================
# ERROR MESSAGES
# =============================================================================

def get_user_friendly_error(error_type: str, details: str = "") -> str:
    """
    Get a user-friendly error message for common error types.
    
    Args:
        error_type: Type of error
        details: Additional details
        
    Returns:
        User-friendly error message
    """
    error_messages = {
        "no_part2": (
            "This PDF does not appear to be a valid specification document. "
            "Could not find the standard 'PART 2 - PRODUCTS' section or equivalent. "
            "Please upload a construction specification file with the required structure."
        ),
        "empty_pdf": (
            "Could not extract any text from this PDF. "
            "The file may be corrupted, image-only, or password protected. "
            "Please ensure the PDF contains selectable text."
        ),
        "extraction_failed": (
            "An error occurred while extracting products from this document. "
            "Please try again or contact support if the issue persists."
        ),
        "timeout": (
            "The extraction process took too long to complete. "
            "This may be due to the document's size or complexity. "
            "Please try with a smaller document or contact support."
        ),
        "invalid_file": (
            "The uploaded file is not a valid PDF document. "
            "Please upload a PDF file with a .pdf extension."
        ),
        "api_error": (
            "A temporary service error occurred. "
            "Please try again in a few moments."
        ),
        "rate_limit": (
            "You have reached the maximum number of extractions allowed. "
            "Please wait a moment before trying again."
        )
    }
    
    base_message = error_messages.get(error_type, 
        "An unexpected error occurred. Please try again or contact support."
    )
    
    if details:
        return f"{base_message} ({details})"
    return base_message


# =============================================================================
# RESPONSE BUILDER
# =============================================================================

def build_extraction_response(
    products: List[Dict[str, Any]],
    detection_result: Dict[str, Any],
    processing_time: float,
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    total_cost: float,
    part2_text: str = "",
    file_size_bytes: int = 0,
    truncated: bool = False,
    original_length: int = 0,
    truncated_length: int = 0,
    max_length: int = 0,
    retry_attempts: int = 1
) -> Dict[str, Any]:
    """
    Build a complete extraction response with all warnings and metadata.
    
    Args:
        products: List of extracted products
        detection_result: Section detection result
        processing_time: Total processing time in seconds
        model_name: Name of the model used
        input_tokens: Input token count
        output_tokens: Output token count
        total_cost: Total cost of extraction
        part2_text: Extracted text for viewer
        file_size_bytes: Original file size
        truncated: Whether content was truncated
        original_length: Original content length
        truncated_length: Truncated content length
        max_length: Maximum allowed length
        retry_attempts: Number of retry attempts used
        
    Returns:
        Complete response dictionary
    """
    warnings = []
    notices = []
    
    product_count = len(products)
    file_size_mb = file_size_bytes / (1024 * 1024) if file_size_bytes > 0 else 0
    
    # Generate warnings based on conditions
    
    # 1. Product count warning
    product_warning = generate_large_product_count_warning(product_count)
    if product_warning:
        warnings.append(product_warning)
    
    # 2. Non-standard section warning
    if detection_result.get("products_section") != "PART_2":
        section_warning = generate_non_standard_section_warning(
            detection_result.get("products_section", "UNKNOWN")
        )
        warnings.append(section_warning)
    
    # 3. Low confidence warning
    confidence = detection_result.get("confidence", 1.0)
    confidence_warning = generate_low_confidence_warning(
        confidence,
        detection_result.get("detection_method", "unknown")
    )
    if confidence_warning:
        warnings.append(confidence_warning)
    
    # 4. Fallback detection warning
    if detection_result.get("detection_method") == "fallback":
        fallback_warning = generate_fallback_detection_warning(
            detection_result.get("reasoning", "Unknown reason")
        )
        warnings.append(fallback_warning)
    
    # 5. File size warning
    size_warning = generate_file_size_warning(file_size_mb)
    if size_warning:
        warnings.append(size_warning)
    
    # 6. Truncation warning
    if truncated and original_length > 0:
        truncation_warning = generate_truncated_content_warning(
            original_length,
            truncated_length,
            max_length
        )
        warnings.append(truncation_warning)
    
    # 7. Retry warning (silent)
    if retry_attempts > 1:
        retry_warning = generate_retry_warning(retry_attempts, 3)
        warnings.append(retry_warning)
    
    # Build user-facing notices (only warnings with user messages)
    for warning in warnings:
        if warning.user_message:
            notices.append(warning.user_message)
    
    # Build response
    response = {
        "products": products,
        "products_count": product_count,
        "part2_text": part2_text,
        "model_name": model_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_cost": total_cost,
        "processing_time": round(processing_time, 2),
        
        # Section detection info
        "section_detection": {
            "products_section": detection_result.get("products_section", "PART_2"),
            "confidence": detection_result.get("confidence", 1.0),
            "detection_method": detection_result.get("detection_method", "rule_based"),
            "reasoning": detection_result.get("reasoning", ""),
            "sections_found": detection_result.get("sections_found", {})
        },
        
        # Warnings and notices
        "warnings": [w.to_dict() for w in warnings],
        "notices": notices,
        "has_warnings": len(warnings) > 0,
        "has_critical_warnings": any(w.severity == WarningSeverity.CRITICAL for w in warnings),
        
        # Metadata
        "metadata": {
            "file_size_mb": round(file_size_mb, 2),
            "truncated": truncated,
            "retry_attempts": retry_attempts
        }
    }
    
    return response


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_warnings_for_log(warnings: List[ExtractionWarning]) -> str:
    """Format warnings for logging."""
    if not warnings:
        return "No warnings"
    
    return " | ".join([
        f"[{w.severity.value.upper()}] {w.message}"
        for w in warnings
    ])


def should_show_warning_banner(warnings: List[ExtractionWarning]) -> bool:
    """Determine if a warning banner should be shown in the UI."""
    return any(
        w.severity in [WarningSeverity.WARNING, WarningSeverity.CRITICAL]
        and w.user_message
        for w in warnings
    )


def get_primary_warning_message(warnings: List[ExtractionWarning]) -> Optional[str]:
    """Get the most important warning message to display."""
    # Sort by severity (critical first)
    sorted_warnings = sorted(
        [w for w in warnings if w.user_message],
        key=lambda w: (
            0 if w.severity == WarningSeverity.CRITICAL else
            1 if w.severity == WarningSeverity.WARNING else
            2
        )
    )
    
    return sorted_warnings[0].user_message if sorted_warnings else None