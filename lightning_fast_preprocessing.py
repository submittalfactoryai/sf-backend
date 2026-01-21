import fitz  # PyMuPDF
import os
import re
import logging
import tempfile
import time
import unicodedata
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path

# --- Configuration & Constants ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define flexible regex patterns for PART 2 and PART 3 detection
PART_2_START_REGEX = r"PART\s*[-:]?\s*2\s*[-:]?\s*PRODUCTS?"
PART_3_START_REGEX = r"PART\s*[-:]?\s*3\s*[-:]?\s*EXECUTION"

# --- Pre-compiled Regex Patterns for Lightning-Fast Text Cleaning ---
CONTROL_CHARS_PATTERN = re.compile(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]')
EXCESSIVE_WHITESPACE_PATTERN = re.compile(r'[ \t]+')
EXCESSIVE_NEWLINES_PATTERN = re.compile(r'\n{3,}')
LINE_ENDING_PATTERN = re.compile(r'\r\n|\r')
TRAILING_SPACES_PATTERN = re.compile(r' +\n')
LEADING_SPACES_PATTERN = re.compile(r'\n +')

# Pre-defined character mappings for maximum speed
PROBLEMATIC_CHARS_MAP = str.maketrans({
    '\ufeff': '',       # UTF-8 BOM
    '\ufffe': '',       # UTF-16 BOM (little endian)
    '\uffff': '',       # Invalid Unicode
    '\ufffd': '',       # Unicode replacement character
    '\u00ff': '',       # 'ÿ' - Latin Small Letter Y with Diaeresis
    '\u0080': '',       # Euro sign when misencoded
    '\u0081': '',       # High Octet Preset
    '\u008d': '',       # Reverse Index
    '\u008f': '',       # Single Shift Three
    '\u0090': '',       # Device Control String
    '\u009d': '',       # Operating System Command
    '\u00a0': ' ',      # Non-breaking space → regular space
    '\u2009': ' ',      # Thin space → regular space
    '\u200a': ' ',      # Hair space → regular space
    '\u2007': ' ',      # Figure space → regular space
    '\u2008': ' ',      # Punctuation space → regular space
    '\u200b': '',       # Zero-width space
    '\u200c': '',       # Zero-width non-joiner
    '\u200d': '',       # Zero-width joiner
    '\u2060': '',       # Word joiner
})

# =============================================================================
# NEW: TOC DETECTION HELPERS
# =============================================================================

# Pattern to detect Table of Contents entries (dots followed by page number)
TOC_PATTERNS = [
    r'\.{3,}\s*\d+',           # Multiple dots followed by page number: "PRODUCTS.......... 5"
    r'(\.\s+){3,}\d+',         # Dots with spaces: ". . . . 5"
    r'\.{2,}',                 # Multiple consecutive dots (common in TOC)
]

# Pattern to detect actual section headers (followed by subsections like 2.1, A., etc.)
ACTUAL_SECTION_INDICATORS = [
    r'2\.1\.?\s+[A-Z]',        # 2.1 followed by letter (like "2.1 ACCEPTABLE")
    r'\n\s*A\.\s+[A-Z]',       # Lettered list items
    r'\n\s*1\.\s+[A-Z]',       # Numbered list items
]


def is_toc_entry(line: str, context_after: str = "") -> bool:
    """
    Determine if a line containing "PART 2" is a Table of Contents entry
    rather than an actual section header.
    
    Args:
        line: The line containing "PART 2 - PRODUCTS" or similar
        context_after: The text following this line (next 200-500 chars)
        
    Returns:
        True if this appears to be a TOC entry, False if it's actual content
    """
    # Check for TOC patterns in the line itself
    for pattern in TOC_PATTERNS:
        if re.search(pattern, line):
            return True
    
    # If the line is very short (just the header), check what follows
    if len(line.strip()) < 50:
        # Check if context_after contains actual section content
        for pattern in ACTUAL_SECTION_INDICATORS:
            if re.search(pattern, context_after, re.IGNORECASE):
                return False  # This is actual content
    
    # If the line ends with a number (page reference), it's likely TOC
    if re.search(r'\d+\s*$', line.strip()):
        return True
    
    return False


def is_toc_page(page_text: str) -> bool:
    """
    Determine if a page is primarily a Table of Contents page.
    
    Args:
        page_text: Full text of the page
        
    Returns:
        True if this appears to be a TOC page
    """
    lines = page_text.split('\n')
    
    # Count lines that look like TOC entries (dots + numbers)
    toc_line_count = 0
    total_non_empty_lines = 0
    
    for line in lines:
        if line.strip():
            total_non_empty_lines += 1
            # TOC entries typically have: "Section Name .......... Page#"
            if re.search(r'\.{3,}\s*\d+', line) or re.search(r'(\.\s+){3,}', line):
                toc_line_count += 1
    
    # If more than 30% of lines look like TOC entries, it's probably a TOC page
    if total_non_empty_lines > 0 and (toc_line_count / total_non_empty_lines) > 0.3:
        return True
    
    # Also check for common TOC headers
    toc_headers = [
        r'table\s+of\s+contents',
        r'^contents$',
    ]
    for header in toc_headers:
        if re.search(header, page_text, re.IGNORECASE | re.MULTILINE):
            return True
    
    return False


# =============================================================================
# LIGHTNING-FAST PDF TEXT EXTRACTION (FIXED)
# =============================================================================

def lightning_fast_check_part2_exists(pdf_path: str) -> bool:
    """
    Lightning-fast check if "PART 2 - PRODUCTS" section exists in the PDF.
    FIXED: Now skips TOC entries to find actual PART 2 content.
    
    Args:
        pdf_path: The path to the PDF file.
        
    Returns:
        True if PART 2 - PRODUCTS is found (actual content, not TOC), False otherwise.
    """
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file not found at: {pdf_path}")
        return False

    try:
        with fitz.open(pdf_path) as doc:
            logging.info(f"⚡ Fast checking PART 2 in {doc.page_count} pages: {os.path.basename(pdf_path)}")
            
            # Compile regex once for speed
            part2_regex = re.compile(PART_2_START_REGEX, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            
            # Process pages to find ACTUAL PART 2 content (not TOC)
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text("text")
                
                # Skip if this is a TOC page
                if is_toc_page(page_text):
                    logging.debug(f"⚡ Skipping TOC page {page_num + 1}")
                    continue
                
                # Quick search on each page
                match = part2_regex.search(page_text)
                if match:
                    # Get the matched line and context
                    matched_line = page_text[match.start():match.end() + 100].split('\n')[0]
                    context_after = page_text[match.end():match.end() + 500]
                    
                    # Check if this is a TOC entry
                    if not is_toc_entry(matched_line, context_after):
                        logging.info(f"⚡ PART 2 found on page {page_num + 1} (actual content)")
                        return True
                    else:
                        logging.debug(f"⚡ Skipping TOC entry on page {page_num + 1}")
            
            logging.warning(f"⚡ PART 2 (actual content) not found in {os.path.basename(pdf_path)}")
            return False

    except fitz.FitzError as e:
        logging.error(f"PyMuPDF error processing {pdf_path}: {e}")
        return False
    except Exception as e:
        logging.error(f"Error checking PART 2 in {pdf_path}: {e}")
        return False


def lightning_fast_find_part2_page(doc, filename: str = "") -> Tuple[Optional[int], str]:
    """
    Find the page with the ACTUAL PART 2 content, skipping TOC entries.
    
    Args:
        doc: PyMuPDF document object
        filename: Original filename for logging
        
    Returns:
        Tuple of (page_number, match_type) where match_type is 'exact', 'fallback', or 'not_found'
    """
    part2_regex = re.compile(PART_2_START_REGEX, re.IGNORECASE | re.MULTILINE | re.DOTALL)
    
    # First pass: Find all pages with PART 2
    all_matches = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text("text")
        
        match = part2_regex.search(page_text)
        if match:
            matched_line = page_text[match.start():match.end() + 100].split('\n')[0]
            context_after = page_text[match.end():match.end() + 500]
            
            is_toc = is_toc_entry(matched_line, context_after) or is_toc_page(page_text)
            
            all_matches.append({
                'page_num': page_num,
                'matched_line': matched_line,
                'is_toc': is_toc,
                'match_start': match.start()
            })
    
    logging.info(f"⚡ Found {len(all_matches)} PART 2 matches in {filename}")
    
    # Second pass: Find the first NON-TOC match
    for match_info in all_matches:
        if not match_info['is_toc']:
            logging.info(f"⚡ PART 2 found on page {match_info['page_num'] + 1} (actual content)")
            return (match_info['page_num'], 'exact')
        else:
            logging.debug(f"⚡ Skipping TOC entry on page {match_info['page_num'] + 1}")
    
    # If all matches were TOC entries, return None
    if all_matches:
        logging.warning(f"⚡ All PART 2 matches appear to be TOC entries in {filename}")
    
    return (None, 'not_found')


def lightning_fast_extract_text_between_patterns(pdf_path: str, start_pattern_regex: str, end_pattern_regex: str) -> Optional[str]:
    """
    Lightning-fast text extraction between two regex patterns.
    FIXED: Now skips TOC entries to extract from actual PART 2 content.

    Args:
        pdf_path: The path to the PDF file.
        start_pattern_regex: The regex pattern marking the start of the desired text.
        end_pattern_regex: The regex pattern marking the end of the desired text.

    Returns:
        The extracted text between the patterns, or None if patterns aren't found.
    """
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file not found at: {pdf_path}")
        return None

    try:
        start_time = time.time()
        
        with fitz.open(pdf_path) as doc:
            filename = os.path.basename(pdf_path)
            logging.info(f"⚡ Fast extracting from {doc.page_count} pages: {filename}")
            
            # Pre-compile regex patterns for speed
            start_regex = re.compile(start_pattern_regex, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            end_regex = re.compile(end_pattern_regex, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            
            # FIXED: Find the actual PART 2 page (not TOC)
            part2_page_num, match_type = lightning_fast_find_part2_page(doc, filename)
            
            if part2_page_num is None:
                logging.warning(f"⚡ No actual PART 2 content found in {filename}")
                return None
            
            # Extract text starting from the actual PART 2 page
            full_text = ""
            for page_num in range(part2_page_num, doc.page_count):
                page = doc[page_num]
                page_text = page.get_text("text")
                full_text += page_text + "\n"

        # Fast pattern matching
        start_match = start_regex.search(full_text)
        if not start_match:
            logging.warning(f"⚡ Start pattern not found after page {part2_page_num + 1}")
            return None
            
        start_index = start_match.start()
        
        # Search for end pattern after start
        end_match = end_regex.search(full_text, pos=start_index + 1)
        if not end_match:
            logging.warning(f"⚡ End pattern not found, using full text from start")
            end_index = len(full_text)
        else:
            end_index = end_match.start()

        extracted_text = full_text[start_index:end_index].strip()
        
        extraction_time = time.time() - start_time
        logging.info(f"⚡ Extracted {len(extracted_text)} chars in {extraction_time:.2f}s (from page {part2_page_num + 1})")
        
        return extracted_text

    except fitz.FitzError as e:
        logging.error(f"PyMuPDF error processing {pdf_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return None


# =============================================================================
# LIGHTNING-FAST TEXT CLEANING (UNCHANGED)
# =============================================================================

def lightning_fast_analyze_text(text: str) -> Dict[str, int]:
    """
    Lightning-fast text analysis focused on critical metrics only.
    Optimized for maximum speed with minimal processing.
    """
    if not text:
        return {"total_chars": 0, "control_chars": 0, "non_ascii_chars": 0}
    
    total_chars = len(text)
    control_chars = len(CONTROL_CHARS_PATTERN.findall(text))
    non_ascii_chars = sum(1 for c in text if ord(c) > 127)
    
    return {
        "total_chars": total_chars,
        "control_chars": control_chars,
        "non_ascii_chars": non_ascii_chars
    }

def lightning_fast_clean_text(text: str) -> str:
    """
    Ultra-fast text cleaning optimized for maximum speed.
    Applies only the most critical cleaning operations in sequence.
    """
    if not text:
        return ""
    
    original_length = len(text)
    
    # Apply critical cleaning operations in optimal order
    # Step 1: Remove control characters (fastest operation)
    text = CONTROL_CHARS_PATTERN.sub('', text)
    
    # Step 2: Replace problematic characters using pre-built translation map
    text = text.translate(PROBLEMATIC_CHARS_MAP)
    
    # Step 3: Normalize whitespace (single operation for speed)
    text = EXCESSIVE_WHITESPACE_PATTERN.sub(' ', text)
    text = EXCESSIVE_NEWLINES_PATTERN.sub('\n\n', text)
    
    # Step 4: Clean line endings and spaces
    text = LINE_ENDING_PATTERN.sub('\n', text)
    text = TRAILING_SPACES_PATTERN.sub('\n', text)
    
    # Step 5: Final trim
    text = text.strip()
    
    return text

def lightning_fast_text_cleaning(raw_text: str) -> Optional[str]:
    """
    Lightning-fast enterprise-grade text cleaning optimized for speed.
    Main entry point for text cleaning operations.
    """
    if not raw_text:
        logging.warning("Received empty raw_text for cleaning.")
        return None

    start_time = time.time()
    logging.info(f"⚡ Starting lightning-fast text cleaning (length: {len(raw_text)})")
    
    try:
        # Fast initial analysis
        initial_stats = lightning_fast_analyze_text(raw_text)
        
        # Lightning-fast cleaning
        cleaned_text = lightning_fast_clean_text(raw_text)
        
        # Quick final validation
        final_stats = lightning_fast_analyze_text(cleaned_text)
        
        chars_removed = initial_stats["total_chars"] - final_stats["total_chars"]
        cleaning_time = time.time() - start_time
        
        if chars_removed > 0:
            logging.info(f"⚡ Fast cleaning: {chars_removed} chars removed, {len(cleaned_text)} final chars in {cleaning_time:.3f}s")
        else:
            logging.info(f"⚡ Fast cleaning: No changes needed, {len(cleaned_text)} chars in {cleaning_time:.3f}s")
        
        return cleaned_text

    except Exception as e:
        logging.error(f"Error during fast text cleaning: {e}")
        return raw_text  # Return original text if cleaning fails


# =============================================================================
# LIGHTNING-FAST DATA PREPARATION (UNCHANGED)
# =============================================================================

def lightning_fast_prepare_product_data(raw_products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Lightning-fast product data preparation and normalization.
    Optimized for maximum speed with minimal processing overhead.
    """
    if not raw_products:
        return []
    
    start_time = time.time()
    products_list = []
    
    logging.info(f"⚡ Fast processing {len(raw_products)} raw products")
    
    for product_data in raw_products:
        if not isinstance(product_data, dict):
            continue
            
        # Fast tech specs processing
        tech_specs = []
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
        
        # Fast manufacturers processing
        manufacturers = []
        mfrs = product_data.get("manufacturers", {})
        
        if isinstance(mfrs, dict):
            for key, value in mfrs.items():
                if isinstance(value, list):
                    manufacturers.extend(value)
        elif isinstance(mfrs, list):
            manufacturers = mfrs
        
        # Build optimized product structure
        products_list.append({
            "product_name": product_data.get("product_name", ""),
            "technical_specifications": tech_specs,
            "manufacturers": manufacturers,
            "reference": product_data.get("reference", "")
        })
    
    processing_time = time.time() - start_time
    logging.info(f"⚡ Processed {len(products_list)} products in {processing_time:.3f}s")
    
    return products_list


# =============================================================================
# LIGHTNING-FAST FILE OPERATIONS (UNCHANGED)
# =============================================================================

def lightning_fast_file_operations(file_content: bytes, filename: str) -> Tuple[str, str]:
    """
    Lightning-fast file operations for temporary file handling.
    Optimized for maximum speed with efficient I/O operations.
    """
    start_time = time.time()
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_pdf_path = os.path.join(temp_dir, filename)
    
    try:
        # Fast file write
        with open(temp_pdf_path, "wb") as temp_file:
            temp_file.write(file_content)
        
        file_time = time.time() - start_time
        logging.info(f"⚡ Fast file operations completed in {file_time:.3f}s")
        
        return temp_pdf_path, temp_dir
        
    except Exception as e:
        logging.error(f"Error in fast file operations: {e}")
        # Cleanup on error
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
        raise

def lightning_fast_cleanup(temp_dir: str) -> None:
    """
    Lightning-fast cleanup of temporary files and directories.
    """
    if temp_dir and os.path.exists(temp_dir):
        try:
            import shutil
            shutil.rmtree(temp_dir)
            logging.debug(f"⚡ Fast cleanup completed: {temp_dir}")
        except Exception as cleanup_err:
            logging.warning(f"Fast cleanup failed for {temp_dir}: {cleanup_err}")


# =============================================================================
# MAIN LIGHTNING-FAST PREPROCESSING PIPELINE
# =============================================================================

def lightning_fast_preprocessing_pipeline(file_content: bytes, filename: str) -> Dict[str, Any]:
    """
    Complete lightning-fast preprocessing pipeline that processes PDF files
    before LLM processing. This is the main entry point for the fast preprocessing.
    
    FIXED: Now properly skips TOC entries to find actual PART 2 content.
    
    Args:
        file_content: Raw PDF file content as bytes
        filename: Original filename of the PDF
        
    Returns:
        Dictionary containing:
        - cleaned_text: Cleaned text content from PART 2
        - processing_time: Total processing time
        - file_size: Size of the input file
        - success: Boolean indicating success/failure
        - error_message: Error message if failed
    """
    pipeline_start_time = time.time()
    temp_dir = None
    
    try:
        logging.info(f"⚡ Starting lightning-fast preprocessing pipeline for: {filename}")
        
        # Step 1: Lightning-fast file operations
        temp_pdf_path, temp_dir = lightning_fast_file_operations(file_content, filename)
        
        # Step 2: Lightning-fast PART 2 validation (FIXED: now skips TOC)
        part2_present = lightning_fast_check_part2_exists(temp_pdf_path)
        if not part2_present:
            return {
                "success": False,
                "error_message": f"PART 2 - PRODUCTS not found in {filename}",
                "processing_time": time.time() - pipeline_start_time,
                "file_size": len(file_content)
            }
        
        # Step 3: Lightning-fast text extraction (FIXED: now extracts from actual PART 2 page)
        raw_text = lightning_fast_extract_text_between_patterns(
            temp_pdf_path, PART_2_START_REGEX, PART_3_START_REGEX
        )
        
        if not raw_text:
            return {
                "success": False,
                "error_message": f"Failed to extract PART 2 text from {filename}",
                "processing_time": time.time() - pipeline_start_time,
                "file_size": len(file_content)
            }
        
        # Step 4: Lightning-fast text cleaning
        cleaned_text = lightning_fast_text_cleaning(raw_text)
        
        if not cleaned_text:
            return {
                "success": False,
                "error_message": f"Text cleaning failed for {filename}",
                "processing_time": time.time() - pipeline_start_time,
                "file_size": len(file_content)
            }
        
        total_processing_time = time.time() - pipeline_start_time
        
        logging.info(f"⚡ Lightning-fast preprocessing completed in {total_processing_time:.2f}s")
        
        return {
            "success": True,
            "cleaned_text": cleaned_text,
            "processing_time": total_processing_time,
            "file_size": len(file_content),
            "text_length": len(cleaned_text),
            "error_message": None
        }
        
    except Exception as e:
        error_msg = f"Lightning-fast preprocessing failed for {filename}: {str(e)}"
        logging.error(error_msg)
        
        return {
            "success": False,
            "error_message": error_msg,
            "processing_time": time.time() - pipeline_start_time,
            "file_size": len(file_content)
        }
        
    finally:
        # Lightning-fast cleanup
        if temp_dir:
            lightning_fast_cleanup(temp_dir)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_preprocessing_stats() -> Dict[str, str]:
    """
    Get statistics and information about the lightning-fast preprocessing implementation.
    """
    return {
        "implementation": "Lightning-Fast Preprocessing v1.1 (TOC Fix)",
        "optimizations": [
            "Pre-compiled regex patterns",
            "Character translation maps",
            "Efficient memory usage",
            "Minimal I/O operations",
            "Fast file operations",
            "Optimized text processing",
            "TOC detection and skipping"  # NEW
        ],
        "features": [
            "PART 2 - PRODUCTS detection",
            "Text extraction between patterns",
            "Ultra-fast text cleaning",
            "Product data preparation",
            "Temporary file management",
            "Error handling and recovery",
            "TOC page detection",  # NEW
            "Actual content vs TOC differentiation"  # NEW
        ],
        "performance_targets": {
            "text_cleaning": "< 0.1s for typical documents",
            "pdf_extraction": "< 2.0s for typical documents",
            "total_preprocessing": "< 3.0s for typical documents"
        }
    }

def lightning_fast_preprocessing_pipeline_with_patterns(
    file_content: bytes, 
    filename: str,
    start_pattern: Optional[str] = None,
    end_pattern: Optional[str] = None
) -> Dict[str, Any]:
    """
    Lightning-fast preprocessing with CUSTOM section patterns.
    Respects section detection results.
    """
    pipeline_start_time = time.time()
    temp_dir = None
    
    # Default to PART 2 if no patterns provided
    if start_pattern is None:
        start_pattern = PART_2_START_REGEX
    if end_pattern is None:
        end_pattern = PART_3_START_REGEX
    
    try:
        logging.info(f"⚡ Starting preprocessing with custom patterns for: {filename}")
        logging.info(f"   Start pattern: {start_pattern[:50]}...")
        logging.info(f"   End pattern: {end_pattern[:50] if end_pattern else 'None'}...")
        
        # Step 1: Fast file operations
        temp_pdf_path, temp_dir = lightning_fast_file_operations(file_content, filename)
        
        # Step 2: Validate pattern exists (skip TOC)
        start_regex = re.compile(start_pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        
        with fitz.open(temp_pdf_path) as doc:
            pattern_found = False
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text("text")
                
                if is_toc_page(page_text):
                    continue
                
                match = start_regex.search(page_text)
                if match:
                    matched_line = page_text[match.start():match.end() + 100].split('\n')[0]
                    context_after = page_text[match.end():match.end() + 500]
                    
                    if not is_toc_entry(matched_line, context_after):
                        pattern_found = True
                        logging.info(f"⚡ Pattern found on page {page_num + 1}")
                        break
        
        if not pattern_found:
            return {
                "success": False,
                "error_message": f"Pattern not found in {filename}",
                "processing_time": time.time() - pipeline_start_time,
                "file_size": len(file_content)
            }
        
        # Step 3: Extract using custom patterns
        raw_text = lightning_fast_extract_text_between_patterns(
            temp_pdf_path, 
            start_pattern,  # USE DETECTED PATTERN
            end_pattern     # USE DETECTED PATTERN
        )
        
        if not raw_text:
            return {
                "success": False,
                "error_message": f"Failed to extract text from {filename}",
                "processing_time": time.time() - pipeline_start_time,
                "file_size": len(file_content)
            }
        
        # Step 4: Clean text
        cleaned_text = lightning_fast_text_cleaning(raw_text)
        
        if not cleaned_text:
            return {
                "success": False,
                "error_message": f"Text cleaning failed for {filename}",
                "processing_time": time.time() - pipeline_start_time,
                "file_size": len(file_content)
            }
        
        total_processing_time = time.time() - pipeline_start_time
        logging.info(f"⚡ Preprocessing completed in {total_processing_time:.2f}s")
        
        return {
            "success": True,
            "cleaned_text": cleaned_text,
            "processing_time": total_processing_time,
            "file_size": len(file_content),
            "text_length": len(cleaned_text),
            "error_message": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "error_message": f"Preprocessing failed: {str(e)}",
            "processing_time": time.time() - pipeline_start_time,
            "file_size": len(file_content)
        }
    finally:
        if temp_dir:
            lightning_fast_cleanup(temp_dir)


if __name__ == "__main__":
    # Example usage and testing
    print("Lightning-Fast Preprocessing Implementation (FIXED)")
    print("=" * 50)
    
    stats = get_preprocessing_stats()
    print(f"Implementation: {stats['implementation']}")
    print(f"Optimizations: {len(stats['optimizations'])} applied")
    print(f"Features: {len(stats['features'])} implemented")
    
    print("\nThis module contains the FIXED lightning-fast preprocessing logic")
    print("that properly handles Table of Contents pages.")
    print("It now extracts from the ACTUAL PART 2 content, not TOC references.")