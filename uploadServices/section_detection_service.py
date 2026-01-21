
import os
import re
import logging
import json
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CUSTOM EXCEPTIONS FOR CRITICAL ERRORS
# =============================================================================

class SectionDetectionError(Exception):
    """Base exception for section detection errors."""
    pass


class GeminiNotAvailableError(SectionDetectionError):
    """Raised when Gemini library is not installed."""
    pass


class APIKeyNotConfiguredError(SectionDetectionError):
    """Raised when GOOGLE_API_KEY is not set."""
    pass


class ModelNotFoundError(SectionDetectionError):
    """Raised when the specified Gemini model is not found."""
    pass


class GeminiAPIError(SectionDetectionError):
    """Raised when Gemini API returns an error."""
    pass


class PDFExtractionError(SectionDetectionError):
    """Raised when PDF text extraction fails."""
    pass


class InvalidResponseError(SectionDetectionError):
    """Raised when Gemini returns invalid/unparseable response."""
    pass


# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

class ProductSection(Enum):
    """Enum representing possible sections where products can be found."""
    PART_1 = "PART_1"
    PART_2 = "PART_2"
    PART_3 = "PART_3"
    UNKNOWN = "UNKNOWN"
    ENTIRE_DOCUMENT = "ENTIRE_DOCUMENT"
    MULTIPLE_SECTIONS = "MULTIPLE_SECTIONS"


# Section regex patterns for extraction
SECTION_PATTERNS = {
    "PART_1": [
        r"PART\s*[-:]?\s*1\s*[-:]?\s*GENERAL",
        r"PART\s*I\b.*?\bGENERAL\b",
        r"GENERAL\s*\(PART\s*1\)",
        r"PART\s*1\s*[-–—]?\s*GENERAL",
    ],
    "PART_2": [
        r"PART\s*[-:]?\s*2\s*[-:]?\s*PRODUCTS?",
        r"PART\s*II\b.*?\bPRODUCTS?\b",
        r"PRODUCTS?\s*\(PART\s*2\)",
        r"PART\s*2\s*[-–—]?\s*PRODUCTS?",
        r"PART\s*2\b",
    ],
    "PART_3": [
        r"PART\s*[-:]?\s*3\s*[-:]?\s*EXECUTION",
        r"PART\s*III\b.*?\bEXECUTION\b",
        r"EXECUTION\s*\(PART\s*3\)",
        r"PART\s*3\s*[-–—]?\s*EXECUTION",
    ]
}

# Model configuration - USE CORRECT MODEL NAME
DETECTION_MODEL_NAME = os.getenv("SECTION_DETECTION_MODEL", "gemini-2.0-flash")
DETECTION_TIMEOUT_S = int(os.getenv("SECTION_DETECTION_TIMEOUT_S", "120"))
MAX_DOCUMENT_CHARS = int(os.getenv("SECTION_DETECTION_MAX_CHARS", "50000"))


# =============================================================================
# TOC (TABLE OF CONTENTS) DETECTION HELPERS
# =============================================================================

TOC_PATTERNS = [
    r'\.{3,}\s*\d+',           # Multiple dots followed by page number
    r'(\.\s+){3,}\d+',         # Dots with spaces
    r'\.{2,}',                 # Multiple consecutive dots
]


def is_toc_entry(line: str, context_after: str = "") -> bool:
    """Determine if a line containing section header is a TOC entry."""
    for pattern in TOC_PATTERNS:
        if re.search(pattern, line):
            return True
    
    if re.search(r'\d+\s*$', line.strip()):
        return True
    
    return False


def is_toc_page(page_text: str) -> bool:
    """Determine if a page is primarily a Table of Contents page."""
    lines = page_text.split('\n')
    toc_line_count = 0
    total_non_empty_lines = 0
    
    for line in lines:
        if line.strip():
            total_non_empty_lines += 1
            if re.search(r'\.{3,}\s*\d+', line) or re.search(r'(\.\s+){3,}', line):
                toc_line_count += 1
    
    if total_non_empty_lines > 0 and (toc_line_count / total_non_empty_lines) > 0.3:
        return True
    
    toc_headers = [r'table\s+of\s+contents', r'^contents$']
    for header in toc_headers:
        if re.search(header, page_text, re.IGNORECASE | re.MULTILINE):
            return True
    
    return False


# =============================================================================
# FULL DOCUMENT TEXT EXTRACTION
# =============================================================================

def extract_full_document_text(
    pdf_path: str = None, 
    pdf_content: bytes = None, 
    max_chars: int = MAX_DOCUMENT_CHARS,
    skip_toc: bool = True
) -> Tuple[str, Dict[str, Any]]:
    """
    Extract the FULL text from the PDF document for Gemini analysis.
    
    RAISES:
        PDFExtractionError: If text extraction fails
    """
    metadata = {
        "total_pages": 0,
        "extracted_pages": 0,
        "skipped_toc_pages": 0,
        "total_chars": 0,
        "truncated": False
    }
    
    doc = None
    
    try:
        if pdf_content is not None and fitz:
            doc = fitz.open(stream=pdf_content, filetype="pdf")
        elif pdf_path and fitz:
            doc = fitz.open(pdf_path)
        elif pdf_path and PdfReader:
            reader = PdfReader(pdf_path)
            text_parts = []
            total_chars = 0
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                
                if skip_toc and is_toc_page(page_text):
                    metadata["skipped_toc_pages"] += 1
                    continue
                
                text_parts.append(f"\n--- PAGE {page_num + 1} ---\n{page_text}")
                total_chars += len(page_text)
                metadata["extracted_pages"] += 1
                
                if total_chars >= max_chars:
                    metadata["truncated"] = True
                    break
            
            metadata["total_pages"] = len(reader.pages)
            full_text = "\n".join(text_parts)
            metadata["total_chars"] = len(full_text)
            return full_text[:max_chars], metadata
        else:
            raise PDFExtractionError("No suitable PDF library available (fitz or pypdf required)")
        
        if doc:
            text_parts = []
            total_chars = 0
            metadata["total_pages"] = len(doc)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text("text")
                
                if skip_toc and is_toc_page(page_text):
                    metadata["skipped_toc_pages"] += 1
                    logger.debug(f"Skipping TOC page {page_num + 1}")
                    continue
                
                text_parts.append(f"\n--- PAGE {page_num + 1} ---\n{page_text}")
                total_chars += len(page_text)
                metadata["extracted_pages"] += 1
                
                if total_chars >= max_chars:
                    metadata["truncated"] = True
                    break
            
            doc.close()
            full_text = "\n".join(text_parts)
            metadata["total_chars"] = len(full_text)
            
            logger.info(f"Extracted {metadata['extracted_pages']}/{metadata['total_pages']} pages, "
                       f"{metadata['total_chars']} chars (skipped {metadata['skipped_toc_pages']} TOC pages)")
            
            if not full_text.strip():
                raise PDFExtractionError("PDF extraction returned empty text")
            
            return full_text[:max_chars], metadata
            
    except PDFExtractionError:
        raise
    except Exception as e:
        if doc:
            doc.close()
        raise PDFExtractionError(f"Failed to extract text from PDF: {str(e)}")
    
    raise PDFExtractionError("Failed to extract text from PDF")


# =============================================================================
# GEMINI-BASED FULL DOCUMENT ANALYSIS
# =============================================================================

def build_full_document_analysis_prompt(document_text: str) -> str:
    """Build the prompt for Gemini to analyze the FULL document."""
    prompt = f"""You are an expert construction specification document analyzer. Your task is to analyze this COMPLETE document and determine which section(s) contain PRODUCT information.

IMPORTANT: Analyze the ENTIRE document carefully. Products can be in:
- PART 1 - GENERAL (sometimes contains product references)
- PART 2 - PRODUCTS (standard location for products, materials, manufacturers)
- PART 3 - EXECUTION (sometimes contains product specifications for installation)
- Or distributed across multiple sections

WHAT TO LOOK FOR:
1. Product names and model numbers
2. Manufacturer names (e.g., Hubbell, Leviton, Brady, Siemens)
3. Material specifications
4. Technical specifications (dimensions, ratings, capacities)
5. "Acceptable Manufacturers" lists
6. "Basis of Design" products
7. Equipment lists

COMPLETE DOCUMENT:
```
{document_text}
```

Analyze this document and respond with a JSON object containing:

{{
    "primary_products_section": "PART_1" or "PART_2" or "PART_3" or "ENTIRE_DOCUMENT",
    "sections_with_products": ["PART_1", "PART_2", "PART_3"],
    "confidence": 0.95,
    "products_found": [
        {{
            "section": "PART_2",
            "product_types": ["Wiring Devices", "Receptacles", "Switches"],
            "manufacturers_mentioned": ["Hubbell", "Leviton", "Pass & Seymour"],
            "has_specifications": true
        }}
    ],
    "document_structure": {{
        "has_part1": true,
        "has_part2": true,
        "has_part3": true,
        "is_standard_format": true
    }},
    "reasoning": "Detailed explanation of where products are located and why",
    "extraction_recommendation": "PART_2" or "PART_1_AND_PART_2" or "ENTIRE_DOCUMENT"
}}

RULES:
1. Be thorough - scan EVERY section for product information
2. If products are only in PART 2, set primary_products_section to "PART_2"
3. If products are in multiple sections, list ALL of them in sections_with_products
4. If products are scattered throughout, set primary_products_section to "ENTIRE_DOCUMENT"
5. Always provide specific product types and manufacturers you found
6. extraction_recommendation should indicate which section(s) to extract for best results

Respond ONLY with the JSON object, no other text."""

    return prompt


def analyze_document_with_gemini(document_text: str) -> Dict[str, Any]:
    """
    Use Gemini to analyze the FULL document and detect which sections contain products.
    
    RAISES:
        GeminiNotAvailableError: If Gemini library is not installed
        APIKeyNotConfiguredError: If GOOGLE_API_KEY is not set
        ModelNotFoundError: If the specified model is not found
        GeminiAPIError: If Gemini API returns an error
        InvalidResponseError: If response cannot be parsed
    """
    # Check if Gemini is available
    if not genai:
        raise GeminiNotAvailableError(
            "Google Generative AI library is not installed. "
            "Install with: pip install google-generativeai"
        )
    
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise APIKeyNotConfiguredError(
            "GOOGLE_API_KEY environment variable is not configured. "
            "Set it with: export GOOGLE_API_KEY='your-api-key'"
        )
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(DETECTION_MODEL_NAME)
        
        prompt = build_full_document_analysis_prompt(document_text)
        
        generation_config = {
            "max_output_tokens": 2000,
            "temperature": 0.1,
        }
        
        logger.info(f"Sending {len(document_text)} chars to Gemini model '{DETECTION_MODEL_NAME}' for analysis...")
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            request_options={"timeout": DETECTION_TIMEOUT_S}
        )
        
        # Check if response has valid content
        if not response.candidates or not response.candidates[0].content.parts:
            raise GeminiAPIError("Gemini returned empty response with no content")
        
        response_text = response.text.strip()
        
        # Clean up response - remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = re.sub(r'^```(?:json)?\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)
        
        # Parse JSON response
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise InvalidResponseError(
                f"Failed to parse Gemini response as JSON: {str(e)}. "
                f"Response text: {response_text[:500]}..."
            )
        
        # Build result
        result = {
            "primary_products_section": parsed.get("primary_products_section", ProductSection.PART_2.value),
            "sections_with_products": parsed.get("sections_with_products", ["PART_2"]),
            "confidence": parsed.get("confidence", 0.5),
            "products_found": parsed.get("products_found", []),
            "document_structure": parsed.get("document_structure", {}),
            "reasoning": parsed.get("reasoning", ""),
            "extraction_recommendation": parsed.get("extraction_recommendation", "PART_2"),
            "detection_method": "gemini_full_analysis",
            "success": True,
            "error": None
        }
        
        # Log token usage if available
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            result["prompt_tokens"] = getattr(response.usage_metadata, 'prompt_token_count', 0)
            result["completion_tokens"] = getattr(response.usage_metadata, 'candidates_token_count', 0)
            logger.info(f"Gemini tokens - Prompt: {result.get('prompt_tokens', 0)}, "
                       f"Completion: {result.get('completion_tokens', 0)}")
        
        logger.info(f"Gemini analysis complete: Primary section={result['primary_products_section']}, "
                   f"Sections with products={result['sections_with_products']}, "
                   f"Confidence={result['confidence']}")
        
        return result
        
    except GeminiNotAvailableError:
        raise
    except APIKeyNotConfiguredError:
        raise
    except InvalidResponseError:
        raise
    except Exception as e:
        error_str = str(e)
        
        # Check for model not found error
        if "is not found" in error_str or "not found for API version" in error_str:
            raise ModelNotFoundError(
                f"Gemini model '{DETECTION_MODEL_NAME}' not found. "
                f"Error: {error_str}. "
                f"Please check the model name and ensure it's available."
            )
        
        # Check for authentication errors
        if "API key" in error_str.lower() or "authentication" in error_str.lower():
            raise APIKeyNotConfiguredError(f"Gemini API authentication failed: {error_str}")
        
        # Generic API error
        raise GeminiAPIError(f"Gemini API error: {error_str}")


# =============================================================================
# MAIN DETECTION FUNCTION (ALWAYS USES GEMINI - NO FALLBACK)
# =============================================================================

def detect_products_section(
    pdf_path: str = None,
    pdf_content: bytes = None,
    use_llm: bool = True,
    force_llm: bool = True
) -> Dict[str, Any]:
    """
    Detect which section(s) contain products using Gemini FULL document analysis.
    
    This function ALWAYS uses Gemini to analyze the complete document.
    CRITICAL: This function will RAISE EXCEPTIONS for errors - NO SILENT FALLBACK.
    
    RAISES:
        PDFExtractionError: If PDF text extraction fails
        GeminiNotAvailableError: If Gemini library is not installed
        APIKeyNotConfiguredError: If GOOGLE_API_KEY is not set
        ModelNotFoundError: If the specified model is not found
        GeminiAPIError: If Gemini API returns an error
        InvalidResponseError: If response cannot be parsed
    
    Args:
        pdf_path: Path to the PDF file
        pdf_content: Raw PDF content as bytes
        use_llm: Ignored - always uses Gemini
        force_llm: Ignored - always uses Gemini
        
    Returns:
        Dictionary containing detection results
    """
    result = {
        "products_section": ProductSection.PART_2.value,
        "sections_with_products": ["PART_2"],
        "confidence": 0.0,
        "detection_method": "gemini_full_analysis",
        "sections_found": {},
        "warnings": [],
        "success": False,
        "error": None,
        "reasoning": "",
        "use_entire_document": False,
        "extraction_recommendation": "PART_2",
        "products_found": [],
        "document_metadata": {}
    }
    
    # Step 1: Extract FULL document text (skipping TOC)
    # This will RAISE PDFExtractionError if it fails
    logger.info("Step 1: Extracting full document text...")
    document_text, metadata = extract_full_document_text(
        pdf_path=pdf_path, 
        pdf_content=pdf_content,
        skip_toc=True
    )
    
    result["document_metadata"] = metadata
    logger.info(f"Extracted {len(document_text)} chars from {metadata['extracted_pages']} pages")
    
    # Step 2: Analyze with Gemini
    # This will RAISE exceptions for any errors - NO FALLBACK
    logger.info("Step 2: Analyzing document with Gemini...")
    gemini_result = analyze_document_with_gemini(document_text)
    
    # If we get here, Gemini analysis succeeded
    result["products_section"] = gemini_result["primary_products_section"]
    result["sections_with_products"] = gemini_result["sections_with_products"]
    result["confidence"] = gemini_result["confidence"]
    result["detection_method"] = "gemini_full_analysis"
    result["reasoning"] = gemini_result["reasoning"]
    result["extraction_recommendation"] = gemini_result["extraction_recommendation"]
    result["products_found"] = gemini_result["products_found"]
    result["success"] = True
    
    # Determine if we need entire document
    if len(gemini_result["sections_with_products"]) > 1:
        result["use_entire_document"] = True
        result["warnings"].append(
            f"Products found in multiple sections: {gemini_result['sections_with_products']}"
        )
    elif gemini_result["primary_products_section"] == "ENTIRE_DOCUMENT":
        result["use_entire_document"] = True
    
    # Build sections_found for backward compatibility
    doc_structure = gemini_result.get("document_structure", {})
    result["sections_found"] = {
        "PART_1": doc_structure.get("has_part1", False),
        "PART_2": doc_structure.get("has_part2", False),
        "PART_3": doc_structure.get("has_part3", False)
    }
    
    # Token usage
    if gemini_result.get("prompt_tokens"):
        result["prompt_tokens"] = gemini_result["prompt_tokens"]
    if gemini_result.get("completion_tokens"):
        result["completion_tokens"] = gemini_result["completion_tokens"]
    
    logger.info(f"Detection complete: {result['products_section']} "
               f"(confidence: {result['confidence']})")
    
    return result


# =============================================================================
# SECTION TEXT EXTRACTION BASED ON DETECTION
# =============================================================================

def get_section_extraction_patterns(products_section: str) -> Tuple[Optional[str], Optional[str]]:
    """Get the regex patterns to extract text based on detected products section."""
    if products_section == "PART_1":
        start_pattern = r"PART\s*[-:]?\s*1\s*[-:]?\s*GENERAL"
        end_pattern = r"PART\s*[-:]?\s*2"
    elif products_section == "PART_2":
        start_pattern = r"PART\s*[-:]?\s*2\s*[-:]?\s*PRODUCTS?"
        end_pattern = r"PART\s*[-:]?\s*3"
    elif products_section == "PART_3":
        start_pattern = r"PART\s*[-:]?\s*3\s*[-:]?\s*EXECUTION"
        end_pattern = None
    else:
        start_pattern = None
        end_pattern = None
    
    return start_pattern, end_pattern


def get_multi_section_extraction_patterns(sections: List[str]) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """Get extraction patterns for multiple sections."""
    patterns = []
    
    for section in sections:
        start_pattern, end_pattern = get_section_extraction_patterns(section)
        if start_pattern:
            patterns.append((section, start_pattern, end_pattern))
    
    return patterns


# =============================================================================
# MAIN FUNCTION FOR EXTERNAL USE
# =============================================================================

async def detect_and_prepare_extraction(
    pdf_path: str = None,
    pdf_content: bytes = None,
    filename: str = "document.pdf"
) -> Dict[str, Any]:
    """
    Main entry point for section detection.
    
    RAISES EXCEPTIONS FOR CRITICAL ERRORS - caller must handle them.
    
    Raises:
        PDFExtractionError: If PDF text extraction fails
        GeminiNotAvailableError: If Gemini library is not installed
        APIKeyNotConfiguredError: If GOOGLE_API_KEY is not set
        ModelNotFoundError: If the specified model is not found
        GeminiAPIError: If Gemini API returns an error
        InvalidResponseError: If response cannot be parsed
    """
    logger.info(f"=" * 60)
    logger.info(f"Starting FULL DOCUMENT analysis for: {filename}")
    logger.info(f"Using Gemini model: {DETECTION_MODEL_NAME}")
    logger.info(f"=" * 60)
    
    # This will raise exceptions for errors - NO FALLBACK
    detection_result = detect_products_section(
        pdf_path=pdf_path,
        pdf_content=pdf_content
    )
    
    start_pattern, end_pattern = get_section_extraction_patterns(
        detection_result["products_section"]
    )
    
    result = {
        "products_section": detection_result["products_section"],
        "sections_with_products": detection_result.get("sections_with_products", []),
        "start_pattern": start_pattern,
        "end_pattern": end_pattern,
        "warnings": detection_result.get("warnings", []),
        "success": detection_result["success"],
        "confidence": detection_result["confidence"],
        "detection_method": detection_result["detection_method"],
        "reasoning": detection_result.get("reasoning", ""),
        "extraction_recommendation": detection_result.get("extraction_recommendation", "PART_2"),
        "products_found": detection_result.get("products_found", []),
        "detection_details": detection_result,
        "use_entire_document": detection_result.get("use_entire_document", False),
        "document_metadata": detection_result.get("document_metadata", {})
    }
    
    # Log summary
    logger.info(f"=" * 60)
    logger.info(f"DETECTION SUMMARY for {filename}:")
    logger.info(f"  Primary Section: {result['products_section']}")
    logger.info(f"  All Sections with Products: {result['sections_with_products']}")
    logger.info(f"  Confidence: {result['confidence']}")
    logger.info(f"  Extraction Recommendation: {result['extraction_recommendation']}")
    logger.info(f"  Use Entire Document: {result['use_entire_document']}")
    
    if result['products_found']:
        logger.info(f"  Products Found:")
        for pf in result['products_found']:
            logger.info(f"    - {pf.get('section')}: {pf.get('product_types', [])} "
                       f"| Manufacturers: {pf.get('manufacturers_mentioned', [])}")
    
    logger.info(f"=" * 60)
    
    return result


# =============================================================================
# CLI FOR TESTING
# =============================================================================

if __name__ == "__main__":
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(description="Detect products section in a PDF using Gemini")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    async def main():
        try:
            result = await detect_and_prepare_extraction(pdf_path=args.pdf_path)
            print("\n" + "=" * 60)
            print("FINAL RESULT:")
            print("=" * 60)
            print(json.dumps(result, indent=2, default=str))
        except SectionDetectionError as e:
            print(f"\n❌ CRITICAL ERROR: {type(e).__name__}")
            print(f"   {str(e)}")
            exit(1)
    
    asyncio.run(main())