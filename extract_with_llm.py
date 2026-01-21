"""
Extract Product Data from PDF Specifications using LLM
Updated with SOW Section 6 compliant standardized error handling.
"""

import sys
import os
import json
import tempfile
import concurrent.futures
import re
import traceback
import logging
import time
from typing import Optional, Dict, Any, List

# Import only the necessary functions
try:
    from to_text import extract_text_between_patterns, PART_2_START_REGEX, PART_3_START_REGEX
    from clean_text import clean_text
    from retrieve import call_gemini_for_extraction
except ImportError:
    from .to_text import extract_text_between_patterns, PART_2_START_REGEX, PART_3_START_REGEX
    from .clean_text import clean_text
    from .retrieve import call_gemini_for_extraction

import datetime
from pathlib import Path

# Import part2_minify
try:
    from part2_minify import minify_part2_with_llm
except ImportError:
    from .part2_minify import minify_part2_with_llm

# Import standardized error handling
try:
    from error_handlers import (
        ErrorCode,
        ErrorCategory,
        create_error_response,
        classify_llm_error,
        classify_extraction_error,
        SubmittalFactoryException,
        handle_gemini_exception,
        extraction_error,
    )
    ERROR_HANDLING_AVAILABLE = True
except ImportError:
    try:
        from error_handlers import (
            ErrorCode,
            ErrorCategory,
            create_error_response,
            classify_llm_error,
            classify_extraction_error,
            SubmittalFactoryException,
            handle_gemini_exception,
            extraction_error,
        )
        ERROR_HANDLING_AVAILABLE = True
    except ImportError:
        ERROR_HANDLING_AVAILABLE = False
        # Define fallback error codes if error_handlers not available
        class ErrorCode:
            NO_TEXT_CONTENT = "2003"
            NO_PART2_FOUND = "2004"
            EXTRACTION_FAILED = "2005"
            MODEL_NOT_FOUND = "3001"
            API_KEY_INVALID = "3003"
            RATE_LIMITED = "3004"
            LLM_TIMEOUT = "3007"
            LLM_RESPONSE_INVALID = "3006"
            CONTEXT_TOO_LONG = "3005"
            INTERNAL_ERROR = "9001"

# --- Configuration & Constants ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_NAME = "gemini-2.5-flash"
TIMEOUT_SECONDS = 800

MIN_PRODUCTS_ACCEPT = int(os.getenv("MIN_PRODUCTS_ACCEPT", "80"))

# --- Few-Shot Prompt ---
FEW_SHOT_PROMPT = """
### **INSTRUCTION**

From the provided construction specification text, extract all distinct products into a single JSON array. A product is a specific, orderable item, uniquely identified by its name combined with critical technical attributes like type, grade, size, or model number.

Recheck if all the distinct products are extracted. Manufacturers and basis of design are not to be considered as products.

For each product identified, create a JSON object with the following four keys:
1.  `product_name`
2.  `technical_specifications`
3.  `manufacturers`
4.  `reference`

---

### **Core Principle: Product Consolidation**

* A product's identity is defined by its **technical characteristics** (e.g., "Gypsum Board, Type X, 1/2 inch"), **not by its manufacturer**.
* If you encounter the same product mentioned multiple times with different approved manufacturers, you **MUST** consolidate this information into a **single product entry**.
* **Action:** Aggregate all manufacturers for a single product into its `manufacturers` field. **DO NOT** create a new, separate product object for each manufacturer listed.

---

### **Field Instructions**

1.  **`product_name`** `(string)`
    * **Action:** Create a unique, descriptive name by combining the product's common name with its key identifiers (e.g., ASTM standard, type, grade, size). **Crucially, do not include manufacturer names in the `product_name` itself.**
    * **Example:** `"Carbon Steel Bolts and Studs (ASTM A307 Grade A)"`

2.  **`technical_specifications`** `(array of objects)`
    * **Action:** Create an array of simple, flat key-value objects. Each object must represent **one single, individual property**. Make sure that you extract maximum amount of product information from the source text.
    * **CRITICAL RULE - DE-AGGREGATION:** If a source sentence or list describes multiple properties, you **MUST** deconstruct it into separate key-value pairs.
    * **Example:**
        * **Source Text:** `"Door curtain: Galvanized steel, flat profile slats, 0.028 inch thick."`
        * **Correct JSON Output:**
            ```json
            [
              { "Door Curtain Material": "Galvanized steel" },
              { "Door Curtain Slat Profile": "Flat profile slats" },
              { "Door Curtain Slat Thickness": "0.028 inch" }
            ]
            ```
    * **Constraint:** Values **MUST** be simple (string, number, boolean). **DO NOT** use nested objects.

3.  **`manufacturers`** `(object)`
    * **Action:** List company names under `base` and `optional` keys (e.g., `{ "base": ["Manufacturer A"], "optional": ["Manufacturer B"] }`).
    * **Constraint:** If no manufacturers are listed, use an empty object `{}`.

4.  **`reference`** `(string)`
    * **Action:** Provide the original text snippet(s) from which the data was extracted, formatted as plain text.
    * **Constraint:** Preserve the source text's original structure, indentation, and hierarchy using plain text formatting only. also ensure you select entire reference content, if you extract product data from a, b, g points in 1 section then you should select entire reference content from a, to g points in 1 section.

---

### **Final Output Rules**

* **JSON Array Only:** The entire response **MUST** be a valid JSON array. It must start with `[` and end with `]`. Do not add any explanations, notes, or markdown fences (` ```json `).
* **CRITICAL: NO MARKDOWN:** Do NOT wrap your response in ```json``` markdown blocks. Return ONLY the raw JSON array.
* **CRITICAL: NO EXTRA TEXT:** Do NOT include any text before or after the JSON array. The response should start with `[` and end with `]`.
* **Completeness & Accuracy:** Extract **ALL** products and their variations. Use the exact wording from the source text and do not infer information.
* **Plain Text Only:** All fields must contain plain text only. No HTML tags are allowed in any field.
* **JSON Validation:** Ensure your JSON is valid and complete. If you run out of space, prioritize completing the current product object rather than starting a new incomplete one.


"""


def _create_error_result(error_code: str, message: str, details: Optional[Dict] = None) -> Dict[str, Any]:
    """Create a standardized error result dictionary for extraction operations."""
    return {
        "success": False,
        "error_code": error_code,
        "error_message": message,
        "data": {"products": []},
        "details": details or {}
    }


def _classify_llm_error_code(error: Exception) -> str:
    """Classify LLM errors into appropriate error codes."""
    error_str = str(error).lower()
    
    if "404" in error_str or "not found" in error_str:
        return ErrorCode.MODEL_NOT_FOUND if ERROR_HANDLING_AVAILABLE else "3001"
    elif "timeout" in error_str or "deadline" in error_str:
        return ErrorCode.LLM_TIMEOUT if ERROR_HANDLING_AVAILABLE else "3007"
    elif "429" in error_str or "rate limit" in error_str or "quota" in error_str:
        return ErrorCode.RATE_LIMITED if ERROR_HANDLING_AVAILABLE else "3004"
    elif "401" in error_str or "403" in error_str or "unauthorized" in error_str:
        return ErrorCode.API_KEY_INVALID if ERROR_HANDLING_AVAILABLE else "3003"
    elif "token" in error_str and ("limit" in error_str or "exceed" in error_str):
        return ErrorCode.CONTEXT_TOO_LONG if ERROR_HANDLING_AVAILABLE else "3005"
    else:
        return ErrorCode.INTERNAL_ERROR if ERROR_HANDLING_AVAILABLE else "9001"


def _get_user_message_for_error(error_code: str) -> str:
    """Get user-friendly message for error code."""
    messages = {
        "3001": "The AI extraction service is temporarily unavailable. Our team has been notified.",
        "3003": "AI service authentication failed. Please contact support.",
        "3004": "You've reached the temporary request limit. Please wait before trying again.",
        "3005": "The document is too large for AI processing. Please upload a shorter specification.",
        "3006": "Unable to extract product details from the uploaded document. Please ensure product specifications are clearly formatted in the PDF or Product Data Sheet.",
        "3007": "The AI processing took too long. Please try with a smaller document.",
        "2003": "No readable text content was found in the uploaded document.",
        "2004": "Could not locate 'PART 2 - PRODUCTS' section in the document.",
        "2005": "Extraction failed due to document structure issues.",
        "9001": "The system experienced an unexpected error. Please try again.",
    }
    return messages.get(error_code, "An unexpected error occurred. Please try again.")


def _export_prompt(prompt: str, export_dir: Optional[str], pdf_path: Optional[str]) -> str:
    """
    Saves the built prompt to a .txt file and returns the full path.
    Filename format: <pdf_base>__YYYY-MM-DD_HH-MM-SS.txt (sanitized)
    """
    # Resolve export directory
    export_root = Path(export_dir or os.getenv("PROMPT_EXPORT_DIR", "./prompts")).resolve()
    export_root.mkdir(parents=True, exist_ok=True)

    # Build a safe base name from pdf file (if available)
    base = "prompt"
    if pdf_path:
        try:
            base = Path(pdf_path).stem
        except Exception:
            pass

    # Timestamped file
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fname = f"{base}__{ts}.txt"

    # Basic filename sanitization
    safe_fname = re.sub(r"[^A-Za-z0-9._\-]+", "_", fname)
    fpath = export_root / safe_fname

    with open(fpath, "w", encoding="utf-8") as f:
        f.write(prompt)

    logging.info(f"Exported LLM prompt to: {fpath}")
    return str(fpath)


# --- Helper Functions ---
def _get_gemini_api_key() -> str:
    """Retrieves the Gemini API key from environment variables."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logging.error("GOOGLE_API_KEY environment variable not set.")
        raise ValueError("GOOGLE_API_KEY not set in environment.")
    return api_key


def _parse_gemini_response_text(response: Any) -> Optional[str]:
    """Safely extracts text content from various Gemini response formats, including truncated responses."""
    try:
        # First try the standard method, but catch ValueError if it fails
        try:
            if hasattr(response, 'text') and response.text:
                return response.text
        except ValueError as e:
            # This happens when finish_reason=2 (MAX_TOKENS) - continue to extract from parts
            logging.debug(f"response.text access failed: {e}")
            
        # If no text but we have candidates, try to extract from parts
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            
            # Check finish reason to understand why there's no text
            finish_reason = getattr(candidate, 'finish_reason', None)
            if finish_reason == 2:  # MAX_TOKENS
                logging.warning("Response was truncated due to MAX_TOKENS limit")
            
            # Try to extract text from content parts
            if hasattr(candidate, 'content') and candidate.content:
                content = candidate.content
                if hasattr(content, 'parts') and content.parts:
                    text_parts = []
                    for part in content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
                    
                    if text_parts:
                        combined_text = "".join(text_parts)
                        if finish_reason == 2:
                            logging.info(f"Successfully extracted {len(combined_text)} characters from truncated response")
                        return combined_text
        
        # Fallback: try the old method for compatibility
        if hasattr(response, 'parts') and response.parts:
            return "".join(part.text for part in response.parts if hasattr(part, 'text') and part.text)
        
        logging.warning(f"Could not extract text from Gemini response. Response type: {type(response)}")
        logging.warning(f"Response attributes: {[attr for attr in dir(response) if not attr.startswith('_')]}")
        return None
        
    except Exception as e:
        logging.error(f"Error processing Gemini response: {e}")
        try:
            logging.debug(f"Raw Gemini response on error: {response}")
        except Exception:
            logging.debug("Could not log raw Gemini response.")
        return None


def _extract_json_from_text(text: str) -> Optional[str]:
    """Extracts JSON content, potentially embedded in markdown code blocks and attempts to fix common JSON errors only when needed."""
    import re
    
    def fast_json_parse(json_str: str) -> Optional[dict]:
        """Fast JSON parsing without any fixes - returns None if parsing fails."""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    
    def clean_mixed_html_json(json_str: str) -> str:
        """Removes HTML content that appears outside of JSON string values."""
        lines = json_str.split('\n')
        cleaned_lines = []
        in_string = False
        
        for line in lines:
            # Track if we're inside a string value (where HTML is allowed)
            quote_count = line.count('"') - line.count('\\"')
            if quote_count % 2 == 1:
                in_string = not in_string
            
            # If we're not in a string and line contains HTML tags, skip it
            if not in_string and re.search(r'<[^>]+>', line) and not any(char in line for char in ['{', '}', '[', ']', '":', ',']):
                continue
                
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def fix_common_json_errors(json_str: str) -> str:
        """Attempts to fix common JSON syntax errors."""
        # First clean any HTML that appears outside JSON strings
        json_str = clean_mixed_html_json(json_str)
        
        # Remove any trailing commas before } or ]
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        # Fix missing commas between objects/arrays (common LLM error)
        # Look for patterns like }\s*{ (object followed by object)
        json_str = re.sub(r'}\s*{', '},{', json_str)
        # Look for patterns like ]\s*[ (array followed by array)  
        json_str = re.sub(r']\s*\[', '],[', json_str)
        # Look for patterns like }\s*[ (object followed by array)
        json_str = re.sub(r'}\s*\[', '},[', json_str)
        # Look for patterns like ]\s*{ (array followed by object)
        json_str = re.sub(r']\s*{', '],{', json_str)
        
        return json_str

    def attempt_truncated_json_repair(json_str: str) -> str:
        """Attempts to repair truncated JSON by closing incomplete structures."""
        json_str = json_str.strip()
        
        # Count unclosed brackets and braces
        open_braces = json_str.count('{') - json_str.count('}')
        open_brackets = json_str.count('[') - json_str.count(']')
        
        # If we have unclosed structures, try to close them
        if open_braces > 0 or open_brackets > 0:
            logging.info(f"Attempting to repair truncated JSON: {open_braces} unclosed braces, {open_brackets} unclosed brackets")
            
            # Handle severely truncated responses - remove incomplete last entries
            lines = json_str.split('\n')
            cleaned_lines = []
            
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                
                # If this is one of the last few lines, be more aggressive about removing incomplete content
                if i >= len(lines) - 5:  # Last 5 lines
                    # Check for incomplete strings, objects, or values
                    if line_stripped and not line_stripped.endswith((',', '}', ']', '"')):
                        # This looks like a truncated line
                        if '"' in line_stripped and line_stripped.count('"') % 2 == 1:
                            # Incomplete string - remove this and all following lines
                            logging.info(f"Removing truncated string and following content: {line_stripped}")
                            break
                        elif line_stripped.endswith(':'):
                            # Incomplete key-value pair - remove this and following
                            logging.info(f"Removing incomplete key-value pair: {line_stripped}")
                            break
                        elif re.match(r'^[^"]*"[^"]*$', line_stripped):
                            # Line with unclosed quote - remove
                            logging.info(f"Removing line with unclosed quote: {line_stripped}")
                            break
                        elif not any(char in line_stripped for char in ['{', '}', '[', ']', '"', ':']):
                            # Line doesn't contain JSON structural elements - likely truncated content
                            logging.info(f"Removing non-JSON content: {line_stripped}")
                            break
                
                cleaned_lines.append(line)
            
            repaired_json = '\n'.join(cleaned_lines).strip()
            
            # Remove trailing commas that might be left hanging
            repaired_json = re.sub(r',\s*$', '', repaired_json)
            
            # Recount after cleaning
            open_braces = repaired_json.count('{') - repaired_json.count('}')
            open_brackets = repaired_json.count('[') - repaired_json.count(']')
            
            # Now add missing closing brackets/braces
            for _ in range(open_brackets):
                repaired_json += ']'
            for _ in range(open_braces):
                repaired_json += '}'
                
            return repaired_json
        
        return json_str

    def try_advanced_json_fixes(json_str: str) -> Optional[dict]:
        """Attempts advanced JSON fixes including truncation handling - only called when simple fixes fail."""
        
        # Try fixing common errors first
        try:
            fixed_json = fix_common_json_errors(json_str)
            return json.loads(fixed_json)
        except json.JSONDecodeError as e:
            logging.debug(f"JSON parse with common fixes failed: {e}")
        
        # Try truncation repair
        try:
            repaired_json = attempt_truncated_json_repair(json_str)
            if repaired_json != json_str:  # Only try if repair was attempted
                result = json.loads(repaired_json)
                if isinstance(result, list) and len(result) > 0:
                    logging.info(f"Successfully repaired truncated JSON. Recovered {len(result)} products.")
                return result
        except json.JSONDecodeError as e:
            logging.debug(f"Truncated JSON repair failed: {e}")
        except Exception as e:
            logging.debug(f"Truncated JSON repair error: {e}")
        
        # Try a more aggressive approach: find the last complete product object
        try:
            # Look for the last complete product object pattern
            product_pattern = r'"product_name"\s*:\s*"[^"]*"'
            matches = list(re.finditer(product_pattern, json_str))
            
            if matches:
                # Find the start of the last complete product
                last_product_start = matches[-1].start()
                
                # Find the opening brace before this product_name
                text_before = json_str[:last_product_start]
                last_opening_brace = text_before.rfind('{')
                
                if last_opening_brace != -1:
                    # Count braces to find where this object should close
                    brace_count = 0
                    end_pos = last_opening_brace
                    
                    for i, char in enumerate(json_str[last_opening_brace:]):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = last_opening_brace + i
                                break
                    
                    # If we never found a closing brace, truncate before this incomplete product
                    if brace_count > 0:
                        truncated_before_incomplete = json_str[:last_opening_brace].rstrip()
                        if truncated_before_incomplete.endswith(','):
                            truncated_before_incomplete = truncated_before_incomplete.rstrip(',')
                        
                        # Add closing bracket for the main array
                        repaired_json = truncated_before_incomplete + ']'
                        
                        result = json.loads(repaired_json)
                        if isinstance(result, list) and len(result) > 0:
                            logging.warning(f"Successfully repaired JSON by removing incomplete product. Products recovered: {len(result)}")
                        return result
        
        except Exception as e:
            logging.debug(f"Aggressive truncation repair failed: {e}")
        
        # Final attempt: extract all complete products we can find
        try:
            products = []
            # Find all product objects that look complete
            product_objects = re.findall(r'\{[^{}]*"product_name"\s*:\s*"[^"]*"[^{}]*\}', json_str, re.DOTALL)
            
            for product_obj in product_objects:
                try:
                    product = json.loads(product_obj)
                    products.append(product)
                except json.JSONDecodeError:
                    continue
            
            if products:
                logging.warning(f"Emergency extraction: Found {len(products)} complete product objects in truncated response")
                return products
        
        except Exception as e:
            logging.debug(f"Emergency product extraction failed: {e}")
        
        return None

    def extract_markdown_json(text: str) -> Optional[str]:
        """Extract JSON from markdown, handling both complete and truncated cases."""
        # Try normal markdown extraction first (complete case)
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Handle truncated markdown (missing closing ```)
        match = re.search(r'```(?:json)?\s*([\s\S]*)', text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            # Only consider this if it looks like JSON (starts with [ or {)
            if content.startswith(('[', '{')):
                logging.info("Found JSON in truncated markdown block (missing closing ```)")
                return content
        
        return None

    # Strategy: Try fast parsing first, only apply fixes if needed
    
    # First, try to find JSON wrapped in markdown code blocks (including truncated)
    json_content = extract_markdown_json(text)
    if json_content:
        # Fast parse attempt first
        parsed = fast_json_parse(json_content)
        if parsed is not None:
            return json.dumps(parsed)
        
        # Only if fast parse fails, try advanced fixes
        logging.debug("Fast JSON parsing failed for markdown content, attempting fixes...")
        parsed = try_advanced_json_fixes(json_content)
        if parsed is not None:
            return json.dumps(parsed)
    
    # Next, try to find a JSON array or object
    json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text, re.DOTALL)
    if json_match:
        json_content = json_match.group(1).strip()
        
        # Fast parse attempt first
        parsed = fast_json_parse(json_content)
        if parsed is not None:
            return json.dumps(parsed)
            
        # Only if fast parse fails, try advanced fixes
        logging.debug("Fast JSON parsing failed for extracted content, attempting fixes...")
        parsed = try_advanced_json_fixes(json_content)
        if parsed is not None:
            return json.dumps(parsed)
    
    # Finally, check if the entire text is JSON (including potentially truncated)
    stripped_text = text.strip()
    if stripped_text.startswith(("{", "[")):
        # Fast parse attempt first
        parsed = fast_json_parse(stripped_text)
        if parsed is not None:
            return json.dumps(parsed)
            
        # Only if fast parse fails, try advanced fixes
        logging.debug("Fast JSON parsing failed for full text, attempting fixes...")
        parsed = try_advanced_json_fixes(stripped_text)
        if parsed is not None:
            return json.dumps(parsed)
    
    # If nothing worked, log more details
    logging.warning(f"Could not extract valid JSON from text. Length: {len(text)} chars")
    logging.warning(f"Text starts with: {text[:100]}...")
    logging.warning(f"Text ends with: ...{text[-100:]}")
    
    # Try to save the problematic JSON to a temp file for debugging
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(text)
            logging.error(f"Saved problematic LLM response to: {f.name}")
    except Exception:
        pass
    
    return None


def print_precautions():
    """Prints important precautions to stderr for the user."""
    precautions = """
**Precautions for the User (Important Considerations):**

*   **Verification Recommended:** LLM extraction from technical documents can have errors. Always review extracted data for accuracy, especially for critical applications.
*   **Source Document Ambiguity:** The quality of extraction depends on the clarity and structure of the source document. Ambiguous phrasing or unconventional formatting can lead to misinterpretations.
*   **Limited Scope:** This process focuses on the provided 'Part 2' text. It may not interpret complex cross-references or external standards without additional context or capabilities.
*   **No Assumptions:** The model is instructed to extract only explicit information. Missing or implied details will not be included.
"""
    print(precautions, file=sys.stderr)


# --- Simplified Prompt Building Function (No Metadata) ---
def _build_prompt(text_content: str) -> str:
    """
    Constructs the detailed prompt for the Gemini API using only the cleaned text,
    without including metadata.
    
    Args:
        text_content: The cleaned text content from Part 2.
        
    Returns:
        A string prompt for the LLM.
    """
    # Create the example format string separately to avoid f-string issues
    example_format = '[{"product_name":"...", "technical_specifications":[...], "manufacturers":{...}, "reference":"..."}]'
    
    prompt = f"""You are a specialized Construction Specification Analyst. Using the instructions, rules, and examples provided, extract all product information from the specification text below.

                {FEW_SHOT_PROMPT}

                Now analyze the following Part 2 - PRODUCTS section:

                ```text
                {text_content}
                ```

                CRITICAL OUTPUT INSTRUCTIONS:
                - Return ONLY a valid JSON array starting with [ and ending with ]
                - NO markdown code blocks, NO explanations, NO extra text
                - If you approach token limits, complete the current product and close the array with ]
                - Example format: {example_format}
                
                Your response must start immediately with [ and end with ] and contain nothing else but the JSON array:"""    
    return prompt


# --- Simplified Product Extraction Function (No Metadata) with Enhanced Error Handling ---
def extract_product_data(text_content: str, max_retries: int = 3, *, export_prompt_dir: Optional[str] = None, source_pdf_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Extracts product data using Gemini based on provided cleaned text content.
    This version includes standardized error handling per SOW Section 6.
    Includes retry logic for handling extraction failures.

    Args:
        text_content: The cleaned text content from Part 2.
        max_retries: Maximum number of retry attempts (default: 3).

    Returns:
        A dictionary containing:
        - 'data': The extracted product JSON (or [] if failed).
        - 'prompt_tokens': Number of prompt tokens used.
        - 'completion_tokens': Number of completion tokens used.
        - 'model': The model name used.
        - 'retry_attempts': Number of retry attempts made.
        - 'error': Error information if extraction failed (None on success).
    """
    logger = logging.getLogger(__name__)
    result = {
        "data": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "model": MODEL_NAME,
        "retry_attempts": 0,
        "truncated": False,
        "notices": [],
        "error": None  # Added for standardized error handling
    }

    if not text_content:
        logger.warning("No text content provided for product extraction.")
        error_code = ErrorCode.NO_TEXT_CONTENT if ERROR_HANDLING_AVAILABLE else "2003"
        result["error"] = _create_error_result(
            error_code,
            _get_user_message_for_error(error_code),
            {"reason": "Empty text content provided"}
        )
        return result

    # Track total tokens across all attempts
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    for attempt in range(1, max_retries + 1):
        logger.info(f"Product extraction attempt {attempt}/{max_retries}")
        
        try:
            # Build Prompt and Call Gemini for Product Extraction
            prompt = _build_prompt(text_content)
            prompt_length = len(prompt)

            logger.info(f"Calling Gemini API for product extraction using cleaned text (attempt {attempt}), prompt length: {prompt_length}")

            # Call the function which returns a dict with response_text and token counts
            gemini_product_result = call_gemini_for_extraction(prompt, MODEL_NAME)

            # Check for errors from the LLM call
            if gemini_product_result.get("error"):
                llm_error = gemini_product_result["error"]
                error_code = llm_error.get("error_code", "")
                
                logger.error(f"LLM call returned error (attempt {attempt}): {llm_error.get('error_message', 'Unknown error')}")
                
                # DON'T retry for model not found errors - they will never succeed
                if error_code in ["3001", "3003"]:  # MODEL_NOT_FOUND, API_KEY_INVALID
                    logger.error(f"Critical error {error_code} detected, not retrying")
                    result["error"] = llm_error
                    result["prompt_tokens"] = total_prompt_tokens
                    result["completion_tokens"] = total_completion_tokens
                    result["retry_attempts"] = attempt
                    return result
                
                # If this is the last attempt, return the error
                if attempt == max_retries:
                    result["error"] = llm_error
                    result["prompt_tokens"] = total_prompt_tokens
                    result["completion_tokens"] = total_completion_tokens
                    result["retry_attempts"] = attempt
                    return result
                
                # Otherwise, retry with backoff
                retry_delay = 2 ** (attempt - 1)
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue

            # Get text and actual token counts from the result
            raw_text = gemini_product_result.get("response_text")
            attempt_prompt_tokens = gemini_product_result.get("prompt_tokens", 0)
            attempt_completion_tokens = gemini_product_result.get("completion_tokens", 0)
            
            # Accumulate tokens from all attempts
            total_prompt_tokens += attempt_prompt_tokens
            total_completion_tokens += attempt_completion_tokens

            # --- Process and Validate Response ---
            if not raw_text:
                logger.error(f"Failed to get text from product Gemini response (attempt {attempt})")
                if attempt < max_retries:
                    retry_delay = 2 ** (attempt - 1)
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error("All retry attempts exhausted, returning error result")
                    error_code = ErrorCode.LLM_RESPONSE_INVALID if ERROR_HANDLING_AVAILABLE else "3006"
                    result["error"] = _create_error_result(
                        error_code,
                        _get_user_message_for_error(error_code),
                        {"attempt": attempt, "reason": "Empty response from LLM"}
                    )
                    result["prompt_tokens"] = total_prompt_tokens
                    result["completion_tokens"] = total_completion_tokens
                    result["retry_attempts"] = attempt
                    return result

            logger.info(f"LLM response length: {len(raw_text)} characters (attempt {attempt})")
            logger.debug(f"LLM response starts with: {raw_text[:200]}...")
            logger.debug(f"LLM response ends with: ...{raw_text[-200:]}")

            # Check if response was truncated (especially due to MAX_TOKENS)
            response_truncated = False
            if hasattr(gemini_product_result, 'finish_reason') and gemini_product_result.get('finish_reason') == 2:
                response_truncated = True
                logger.warning("Response was truncated due to MAX_TOKENS limit (attempt %s)", attempt)
                result["truncated"] = True
                result["notices"] = ["LLM couldn't extract the full product list due to token limits. Some products may be missing."]

            elif not raw_text.strip().endswith((']', '}')):
                response_truncated = True
                logger.warning("Response appears to be truncated based on ending (attempt %s)", attempt)
                result["truncated"] = True
                result["notices"] = ["LLM couldn't extract the full product list due to token limits. Some products may be missing."]


            json_string = _extract_json_from_text(raw_text)
            if not json_string:
                if response_truncated:
                    logger.error(f"Could not extract JSON from truncated response (attempt {attempt}) - trying emergency extraction")
                    # Try emergency extraction for truncated responses
                    try:
                        # Look for any complete JSON arrays or objects
                        json_pattern = r'(\[[^\[\]]*\]|\{[^{}]*\})'
                        matches = re.findall(json_pattern, raw_text, re.DOTALL)
                        
                        for match in matches:
                            try:
                                parsed = json.loads(match)
                                if isinstance(parsed, list) and len(parsed) > 0:
                                    logger.warning(f"Emergency extraction successful: found {len(parsed)} products in truncated response")
                                    result["data"] = {"products": parsed} if isinstance(parsed, list) else parsed
                                    result["prompt_tokens"] = total_prompt_tokens
                                    result["completion_tokens"] = total_completion_tokens
                                    result["retry_attempts"] = attempt
                                    result["truncated"] = True
                                    result["notices"] = ["LLM couldn't extract the full product list due to token limits. Some products may be missing."]
                                    return result

                            except json.JSONDecodeError:
                                continue
                    except Exception as e:
                        logger.debug(f"Emergency extraction failed: {e}")
                
                logger.error(f"Could not extract JSON structure from product Gemini response (attempt {attempt})")
                logger.error(f"Raw response preview: {raw_text[:500]}...")
                if attempt < max_retries:
                    retry_delay = 2 ** (attempt - 1)
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error("All retry attempts exhausted, returning error result")
                    error_code = ErrorCode.LLM_RESPONSE_INVALID if ERROR_HANDLING_AVAILABLE else "3006"
                    result["error"] = _create_error_result(
                        error_code,
                        _get_user_message_for_error(error_code),
                        {"attempt": attempt, "reason": "Could not parse JSON from response"}
                    )
                    result["prompt_tokens"] = total_prompt_tokens
                    result["completion_tokens"] = total_completion_tokens
                    result["retry_attempts"] = attempt
                    return result

            logger.info(f"Extracted JSON string length: {len(json_string)} characters (attempt {attempt})")

            try:
                json_output = json.loads(json_string)
                
                # Convert the product list to the expected "products" format if needed
                if isinstance(json_output, list):
                    json_output = {"products": json_output}
                
                if not isinstance(json_output, dict) or ('products' not in json_output and not isinstance(json_output, list)):
                    logger.error(f"Product JSON structure validation failed (attempt {attempt}) - expected list or {{'products': [...]}}")
                    if attempt < max_retries:
                        retry_delay = 2 ** (attempt - 1)
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.error("All retry attempts exhausted, returning empty result")
                        result["data"] = {"products": []}
                        result["prompt_tokens"] = total_prompt_tokens
                        result["completion_tokens"] = total_completion_tokens
                        result["retry_attempts"] = attempt
                        return result

                # Check if we actually extracted any products
                products_count = len(json_output.get("products", [])) if isinstance(json_output, dict) else len(json_output)
                logger.info(f"Successfully parsed JSON with {products_count} products (attempt {attempt})")

                # If no products were extracted, retry (unless it's the last attempt)
                if products_count == 0:
                    logger.warning(f"No products extracted from response (attempt {attempt})")
                    if attempt < max_retries:
                        retry_delay = 2 ** (attempt - 1)
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.warning("All retry attempts exhausted, but returning empty products result")
                        result["data"] = json_output
                        result["prompt_tokens"] = total_prompt_tokens
                        result["completion_tokens"] = total_completion_tokens
                        result["retry_attempts"] = attempt
                        return result

                # Success! We have products
                if response_truncated:
                    logger.warning("Successfully extracted %s products from TRUNCATED response on attempt %s", products_count, attempt)
                    logger.warning("Note: Response was truncated due to MAX_TOKENS limit - some products may be missing")
                    result["truncated"] = True
                    if not result.get("notices"):
                        result["notices"] = ["LLM couldn't extract the full product list due to token limits. Some products may be missing."]

                else:
                    logger.info(f"Successfully extracted {products_count} products on attempt {attempt}")
                result["data"] = json_output
                result["prompt_tokens"] = total_prompt_tokens
                result["completion_tokens"] = total_completion_tokens
                result["retry_attempts"] = attempt
                return result

            except json.JSONDecodeError as json_err:
                logger.error(f"Extracted product text is not valid JSON (attempt {attempt}): {json_err}")
                if attempt < max_retries:
                    retry_delay = 2 ** (attempt - 1)
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error("All retry attempts exhausted due to JSON decode errors")
                    error_code = ErrorCode.LLM_RESPONSE_INVALID if ERROR_HANDLING_AVAILABLE else "3006"
                    result["error"] = _create_error_result(
                        error_code,
                        _get_user_message_for_error(error_code),
                        {"attempt": attempt, "reason": f"JSON decode error: {str(json_err)}"}
                    )
                    result["prompt_tokens"] = total_prompt_tokens
                    result["completion_tokens"] = total_completion_tokens
                    result["retry_attempts"] = attempt
                    return result

        except Exception as e:
            logger.error(f"An unexpected error occurred in extract_product_data (attempt {attempt}): {e}")
            traceback.print_exc()
            
            # Classify the error
            error_code = _classify_llm_error_code(e)
            
            if attempt < max_retries:
                retry_delay = 2 ** (attempt - 1)
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                logger.error("All retry attempts exhausted due to unexpected errors")
                result["error"] = _create_error_result(
                    error_code,
                    _get_user_message_for_error(error_code),
                    {"attempt": attempt, "technical": str(e), "type": type(e).__name__}
                )
                result["data"] = []
                result["prompt_tokens"] = total_prompt_tokens
                result["completion_tokens"] = total_completion_tokens
                result["retry_attempts"] = attempt
                return result

    # This should never be reached, but just in case
    logger.error("Unexpected exit from retry loop")
    result["prompt_tokens"] = total_prompt_tokens
    result["completion_tokens"] = total_completion_tokens
    result["retry_attempts"] = max_retries
    return result


# --- Simplified Main Function (No Metadata) ---
def main(pdf_path: str) -> Optional[Dict[str, Any]]:
    """
    Main function for standalone execution.
    Simplified version that doesn't use metadata.
    """
    logger = logging.getLogger(__name__)

    # Initialize simplified result structure (no metadata)
    combined_result = {
        "data": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "model": MODEL_NAME,
        "error": None
    }

    try:
        # 1. Extract Relevant Text Content (Part 2)
        raw_text = extract_text_between_patterns(pdf_path, PART_2_START_REGEX, PART_3_START_REGEX)
        if not raw_text:
            logger.warning(f"Failed to extract relevant text (Part 2) from {pdf_path}. Cannot perform product extraction.")
            error_code = ErrorCode.NO_PART2_FOUND if ERROR_HANDLING_AVAILABLE else "2004"
            combined_result["error"] = _create_error_result(
                error_code,
                _get_user_message_for_error(error_code),
                {"pdf_path": pdf_path}
            )
            return combined_result

        # 2. Clean Text 
        cleaned_text = clean_text(raw_text)
        if not cleaned_text:
            logger.warning(f"Cleaning text resulted in empty content for {pdf_path}. Cannot perform product extraction.")
            error_code = ErrorCode.NO_TEXT_CONTENT if ERROR_HANDLING_AVAILABLE else "2003"
            combined_result["error"] = _create_error_result(
                error_code,
                _get_user_message_for_error(error_code),
                {"pdf_path": pdf_path}
            )
            return combined_result
        
        # 3. Minify Part 2 text
        minified_text, used_fallback = minify_part2_with_llm(
            cleaned_text,
            export_prefix=Path(pdf_path).stem
        )
        logger.info(
            f"Minified PART 2 length: {len(minified_text)} chars "
            f"({'fallback' if used_fallback else 'llm'})"
        )

        # 4. Extract products
        product_extraction_result = extract_product_data(minified_text)

        # Check for extraction errors
        if product_extraction_result.get("error"):
            combined_result["error"] = product_extraction_result["error"]
            combined_result["prompt_tokens"] = product_extraction_result.get("prompt_tokens", 0)
            combined_result["completion_tokens"] = product_extraction_result.get("completion_tokens", 0)
            return combined_result

        # Store product data and token info in combined result
        combined_result["data"] = product_extraction_result.get("data", [])
        combined_result["prompt_tokens"] = product_extraction_result.get("prompt_tokens", 0)
        combined_result["completion_tokens"] = product_extraction_result.get("completion_tokens", 0)
        
        # Copy over any notices/warnings
        if product_extraction_result.get("truncated"):
            combined_result["truncated"] = True
        if product_extraction_result.get("notices"):
            combined_result["notices"] = product_extraction_result["notices"]
            
        logger.info(f"Product extraction used {combined_result['prompt_tokens'] + combined_result['completion_tokens']} tokens (Model: {combined_result['model']}).")

    except Exception as e:
        logger.error(f"An unexpected error occurred in extract_with_llm.main: {e}")
        traceback.print_exc()
        error_code = _classify_llm_error_code(e)
        combined_result["error"] = _create_error_result(
            error_code,
            _get_user_message_for_error(error_code),
            {"technical": str(e), "type": type(e).__name__}
        )
        return combined_result

    return combined_result


# --- Direct Execution ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_with_llm.py <pdf_path>")
        sys.exit(1)
    print_precautions()
    pdf_path = sys.argv[1]
    result = main(pdf_path)
    if result:
        if result.get("error"):
            print(json.dumps(result["error"], indent=2))
            sys.exit(1)
        else:
            print(json.dumps(result["data"], indent=2))
    else:
        print("Error: Processing failed.")
        sys.exit(1)