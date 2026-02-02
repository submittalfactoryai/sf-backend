#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Product Extraction Module with Standardized Error Handling
Extracts PART 2 – PRODUCTS from PDF specifications using Gemini AI.

Updated with SOW Section 6 compliant error handling.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types
from dotenv import load_dotenv

# optional: for local PDF text extraction (no Pinecone)
try:
    from pypdf import PdfReader
    HAVE_PYPDF = True
except Exception:
    HAVE_PYPDF = False

# ------------------------------------------------------------------------------
# PATHS & LOCAL IMPORTS (keep your metadata extractor as-is)
# ------------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from metadata import extract_metadata_dict
except ImportError:
    try:
        from metadata import extract_metadata_dict
    except ImportError:
        extract_metadata_dict = None  # handled later

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
            PDF_NOT_READABLE = "2001"
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

# ------------------------------------------------------------------------------
# LOGGING (advanced, per-step timings + request correlation id)
# ------------------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k in ("req_id", "step", "duration_ms", "event", "details", "error_code"):
            if hasattr(record, k) and getattr(record, k) is not None:
                payload[k] = getattr(record, k)
        return json.dumps(payload, ensure_ascii=False)

def _log(logger: logging.Logger, level: int, msg: str, **kv):
    extra = {k: v for k, v in kv.items() if v is not None}
    logger.log(level, msg, extra=extra)

@contextmanager
def timed_step(logger: logging.Logger, step: str):
    t0 = time.perf_counter()
    _log(logger, logging.INFO, f"START: {step}", step=step, event="start")
    try:
        yield
    finally:
        dt = int((time.perf_counter() - t0) * 1000)
        _log(logger, logging.INFO, f"END: {step} ({dt} ms)", step=step, event="end", duration_ms=dt)

def build_logger(verbosity: int = 0, req_id: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("products_extractor")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        for h in logger.handlers[:]:
            logger.removeHandler(h)

    # human-readable stdout
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO if verbosity == 0 else logging.DEBUG)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)

    # json logs to stderr (for collectors)
    jh = logging.StreamHandler(sys.stderr)
    jh.setLevel(logging.DEBUG)
    jh.setFormatter(JsonFormatter())
    logger.addHandler(jh)

    # store req_id on the logger object
    logger.req_id = req_id or str(uuid.uuid4())  # type: ignore[attr-defined]
    return logger

# ------------------------------------------------------------------------------
# ENV / CONFIG
# ------------------------------------------------------------------------------
load_dotenv()

# Vertex AI Configuration
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "submittalfactoryai")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

# Set credentials path for Google Cloud SDK
if GOOGLE_APPLICATION_CREDENTIALS:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

# char guard before sending to LLM
MAX_SECTION_CHARS = int(os.getenv("MAX_SECTION_CHARS", "180000"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "65000"))  # Vertex AI max is 65536
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
REQUEST_TIMEOUT_S = int(os.getenv("REQUEST_TIMEOUT_S", "18000000"))
RETRIES = int(os.getenv("RETRIES", "1"))  # 1 retry (total 2 attempts)
RETRY_BACKOFF_S = float(os.getenv("RETRY_BACKOFF_S", "2.0"))

# Initialize Vertex AI client
def get_vertex_client():
    """Get or create Vertex AI client."""
    return genai.Client(
        vertexai=True,
        project=GOOGLE_CLOUD_PROJECT,
        location=GOOGLE_CLOUD_LOCATION
    )

# ------------------------------------------------------------------------------
# PART 2 – PRODUCTS isolation (no web, no Pinecone)
# ------------------------------------------------------------------------------
PART2_PATTERNS = [
    r"\bPART\s*2\b.*?\bPRODUCTS\b",
    r"\bPART\s*II\b.*?\bPRODUCTS\b",
    r"\bPRODUCTS\s*\(PART\s*2\)\b",
    r"\bPART\s*2\s*[-–—]?\s*PRODUCTS\b",
]
PART3_PATTERNS = [
    r"\bPART\s*3\b.*?\bEXECUTION\b",
    r"\bPART\s*III\b.*?\bEXECUTION\b",
    r"\bEXECUTION\s*\(PART\s*3\)\b",
    r"\bPART\s*3\s*[-–—]?\s*EXECUTION\b",
]


def _create_error_result(error_code: str, message: str, details: Optional[Dict] = None) -> Dict[str, Any]:
    """Create a standardized error result dictionary."""
    return {
        "success": False,
        "error_code": error_code,
        "error_message": message,
        "products": [],
        "details": details or {}
    }


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


def _read_pdf_text(logger: logging.Logger, pdf_path: str) -> Tuple[str, Optional[Dict]]:
    """
    Read text from PDF file with standardized error handling.
    
    Returns:
        Tuple of (text_content, error_dict). If successful, error_dict is None.
    """
    if not HAVE_PYPDF:
        _log(logger, logging.WARNING, "pypdf not installed; cannot extract PDF text.", 
             error_code=ErrorCode.PDF_NOT_READABLE if ERROR_HANDLING_AVAILABLE else "2001")
        return "", _create_error_result(
            ErrorCode.PDF_NOT_READABLE if ERROR_HANDLING_AVAILABLE else "2001",
            "PDF processing library not available. Please contact support.",
            {"technical": "pypdf not installed"}
        )
    
    with timed_step(logger, "read_pdf"):
        if not os.path.isfile(pdf_path):
            _log(logger, logging.ERROR, f"PDF not found: {pdf_path}",
                 error_code=ErrorCode.PDF_NOT_READABLE if ERROR_HANDLING_AVAILABLE else "2001")
            return "", _create_error_result(
                ErrorCode.PDF_NOT_READABLE if ERROR_HANDLING_AVAILABLE else "2001",
                "The uploaded file could not be found or accessed.",
                {"path": pdf_path}
            )
        
        try:
            size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            _log(logger, logging.INFO, "PDF stats", details={"path": pdf_path, "size_mb": round(size_mb, 3)})
        except Exception:
            pass
        
        try:
            reader = PdfReader(pdf_path)
            pages = len(reader.pages)
            _log(logger, logging.INFO, "PDF page count", details={"pages": pages})
            
            buf: List[str] = []
            failed_pages = []
            for i in range(pages):
                try:
                    buf.append(reader.pages[i].extract_text() or "")
                except Exception as e:
                    _log(logger, logging.WARNING, f"page {i} extract_text failed: {e}")
                    failed_pages.append(i)
            
            text = "\n".join(buf)
            _log(logger, logging.DEBUG, "Raw text length", details={"chars": len(text)})
            
            if not text.strip():
                _log(logger, logging.ERROR, "No text content extracted from PDF",
                     error_code=ErrorCode.NO_TEXT_CONTENT if ERROR_HANDLING_AVAILABLE else "2003")
                return "", _create_error_result(
                    ErrorCode.NO_TEXT_CONTENT if ERROR_HANDLING_AVAILABLE else "2003",
                    "No readable text content was found in the uploaded document. The PDF may be scanned or image-based.",
                    {"pages": pages, "failed_pages": failed_pages}
                )
            
            return text, None
            
        except Exception as e:
            error_str = str(e).lower()
            _log(logger, logging.ERROR, f"PDF read error: {e}", 
                 error_code=ErrorCode.PDF_NOT_READABLE if ERROR_HANDLING_AVAILABLE else "2001")
            
            # Classify the error
            if "encrypted" in error_str or "password" in error_str:
                return "", _create_error_result(
                    ErrorCode.PDF_NOT_READABLE if ERROR_HANDLING_AVAILABLE else "2001",
                    "The PDF appears to be password-protected or encrypted.",
                    {"technical": str(e)}
                )
            elif "corrupt" in error_str or "damaged" in error_str:
                return "", _create_error_result(
                    ErrorCode.PDF_NOT_READABLE if ERROR_HANDLING_AVAILABLE else "2001",
                    "The uploaded file appears to be corrupted and cannot be processed.",
                    {"technical": str(e)}
                )
            else:
                return "", _create_error_result(
                    ErrorCode.PDF_NOT_READABLE if ERROR_HANDLING_AVAILABLE else "2001",
                    "The system's extraction engine could not read this PDF. Please ensure it contains selectable text.",
                    {"technical": str(e)}
                )


def _find_products_span(text: str) -> Optional[Tuple[int, int]]:
    starts = []
    for pat in PART2_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            starts.append(m.start())
    if not starts:
        return None
    start = min(starts)
    end_candidates = []
    for pat in PART3_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE | re.DOTALL):
            if m.start() > start:
                end_candidates.append(m.start())
    end = min(end_candidates) if end_candidates else len(text)
    return (start, end)


def _isolate_products(logger: logging.Logger, full_text: str) -> Tuple[str, Optional[Dict]]:
    """
    Isolate PART 2 - PRODUCTS section with standardized error handling.
    
    Returns:
        Tuple of (section_text, error_dict). If successful but using fallback, error_dict is None.
    """
    with timed_step(logger, "isolate_part2_products"):
        span = _find_products_span(full_text)
        if not span:
            _log(logger, logging.WARNING, "Could not find clear PART 2 – PRODUCTS. Using heuristic middle segment.",
                 error_code=ErrorCode.NO_PART2_FOUND if ERROR_HANDLING_AVAILABLE else "2004")
            n = len(full_text)
            seg = full_text[n//3: 2*n//3]
            # Return with a warning but not an error - we can still try to extract
            return seg[:MAX_SECTION_CHARS], None
        
        start, end = span
        section = full_text[start:end]
        if len(section) > MAX_SECTION_CHARS:
            section = section[:MAX_SECTION_CHARS]
            _log(logger, logging.INFO, "Section truncated to cap", details={"cap_chars": MAX_SECTION_CHARS})
        _log(logger, logging.INFO, "Section length", details={"chars": len(section)})
        return section, None


# ------------------------------------------------------------------------------
# GEMINI CALL with Standardized Error Handling
# ------------------------------------------------------------------------------
def call_gemini_for_extraction(prompt: str, model_name: str) -> Dict[str, Any]:
    """
    Calls the Gemini API via Vertex AI with the given prompt and model name.
    Returns a dictionary containing the response text, token counts, and error info.
    Enhanced with standardized error handling for SOW Section 6 compliance.
    """
    logger = logging.getLogger("products_extractor")
    response_text = None
    prompt_tokens = 0
    completion_tokens = 0
    error_info = None

    try:
        # Initialize Vertex AI client
        try:
            client = get_vertex_client()
            logger.info(f"Vertex AI client initialized for project: {GOOGLE_CLOUD_PROJECT}")
        except Exception as client_error:
            logger.error(f"Failed to initialize Vertex AI client: {client_error}")
            return {
                "response_text": None, 
                "prompt_tokens": 0, 
                "completion_tokens": 0,
                "error": _create_error_result(
                    ErrorCode.API_KEY_INVALID if ERROR_HANDLING_AVAILABLE else "3003",
                    "AI service authentication failed. Please contact support.",
                    {"technical": f"Vertex AI client init failed: {client_error}"}
                )
            }

        logger.info(f"Sending request to Gemini model: {model_name} (for product extraction)")
        logger.info(f"Prompt length: {len(prompt)} characters")

        # Vertex AI generation config
        generation_config = types.GenerateContentConfig(
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
        )

        # --- retries with detailed error handling ---
        last_err = None
        for attempt in range(1, (RETRIES + 2)):
            try:
                with timed_step(logger, f"gemini_generate_attempt_{attempt}"):
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=generation_config,
                    )
                logger.info("Received response from Gemini API (for product extraction).")
                logger.info(f"Response type: {type(response)}")

                try:
                    has_text = hasattr(response, 'text')
                    has_usage = hasattr(response, 'usage_metadata')
                    logger.info(f"Response has text: {has_text}")
                    logger.info(f"Response has usage_metadata: {has_usage}")
                except Exception as debug_error:
                    logger.warning(f"Could not get basic response info: {debug_error}")

                # analyze finish reason
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason'):
                        finish_reason = candidate.finish_reason
                        logger.info(f"Response finish reason: {finish_reason}")
                        if finish_reason == 1:
                            logger.info("Finish reason 1: STOP - Natural completion")
                        elif finish_reason == 2:
                            logger.warning("Response was truncated due to MAX_TOKENS limit")
                        elif finish_reason == 3:
                            logger.warning("Finish reason 3: SAFETY - Content filtered for safety")
                        elif finish_reason == 4:
                            logger.warning("Finish reason 4: RECITATION - Content filtered for recitation")
                        else:
                            logger.warning(f"Unknown finish reason: {finish_reason}")

                    if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                        logger.info(f"Safety ratings: {candidate.safety_ratings}")
                        for rating in candidate.safety_ratings:
                            if hasattr(rating, 'category') and hasattr(rating, 'probability'):
                                logger.info(f"Safety - {rating.category}: {rating.probability}")
                    else:
                        logger.info("No safety ratings found")

                    if hasattr(candidate, 'content'):
                        content = candidate.content
                        if hasattr(content, 'parts') and content.parts:
                            logger.info(f"Content parts count: {len(content.parts)}")
                            for i, part in enumerate(content.parts):
                                if hasattr(part, 'text'):
                                    part_length = len(part.text) if part.text else 0
                                    logger.info(f"Part {i} length: {part_length} characters")
                else:
                    logger.warning("No candidates found in response")

                # usage
                try:
                    if response.usage_metadata:
                        prompt_tokens = response.usage_metadata.prompt_token_count
                        completion_tokens = response.usage_metadata.candidates_token_count
                        logger.info(f"Product Extraction Gemini Usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}")
                    else:
                        logger.warning("Usage metadata not found in product extraction Gemini response.")
                except Exception as meta_error:
                    logger.warning(f"Could not retrieve product extraction token usage metadata: {meta_error}")

                # parse text
                response_text = _parse_gemini_response_text(response)
                if not response_text:
                    logger.error("Failed to extract text from product Gemini response (response parsing failed)")
                    logger.error(f"Response attributes: {dir(response)}")
                    if hasattr(response, 'candidates'):
                        logger.error(f"Response candidates: {response.candidates}")
                        for i, candidate in enumerate(response.candidates):
                            logger.error(f"Candidate {i}: {candidate}")
                            if hasattr(candidate, 'content'):
                                logger.error(f"Candidate {i} content: {candidate.content}")
                    
                    error_info = _create_error_result(
                        ErrorCode.LLM_RESPONSE_INVALID if ERROR_HANDLING_AVAILABLE else "3006",
                        "Unable to extract product details from the uploaded document. Please ensure product specifications are clearly formatted in the PDF or Product Data Sheet.",
                        {"attempt": attempt}
                    )
                    return {"response_text": None, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "error": error_info}

                if response_text and not response_text.strip().endswith((']', '}')):
                    logger.warning(f"Response appears to be truncated. Length: {len(response_text)} chars")
                    logger.warning(f"Response ends with: ...{response_text[-100:]}")

                # success this attempt
                break

            except ValueError as e:
                logger.error(f"Configuration error for product extraction: {e}")
                error_info = _create_error_result(
                    ErrorCode.MODEL_NOT_FOUND if ERROR_HANDLING_AVAILABLE else "3001",
                    "The AI extraction service is temporarily unavailable. Our team has been notified.",
                    {"technical": str(e)}
                )
                return {"response_text": None, "prompt_tokens": 0, "completion_tokens": 0, "error": error_info}
            
            except Exception as e:
                last_err = e
                es = str(e).lower()
                
                # Classify the error for proper handling
                if "deadline exceeded" in es or "timeout" in es or "504" in es:
                    logger.error(f"API timeout/deadline exceeded during product extraction: {e}")
                    error_code = ErrorCode.LLM_TIMEOUT if ERROR_HANDLING_AVAILABLE else "3007"
                    error_message = "The AI processing took too long. Please try with a smaller document."
                elif "404" in es or "not found" in es:
                    logger.error(f"Model not found error: {e}")
                    error_code = ErrorCode.MODEL_NOT_FOUND if ERROR_HANDLING_AVAILABLE else "3001"
                    error_message = "The AI extraction service is temporarily unavailable. Our team has been notified."
                elif "429" in es or "rate limit" in es or "quota" in es:
                    logger.error(f"Rate limit error: {e}")
                    error_code = "3004"  # RATE_LIMITED
                    error_message = "You've reached the temporary request limit. Please wait before trying again."
                elif "401" in es or "403" in es or "unauthorized" in es:
                    logger.error(f"Authentication error: {e}")
                    error_code = ErrorCode.API_KEY_INVALID if ERROR_HANDLING_AVAILABLE else "3003"
                    error_message = "AI service authentication failed. Please contact support."
                else:
                    logger.error(f"Unexpected error during API call: {e}", exc_info=True)
                    error_code = ErrorCode.INTERNAL_ERROR if ERROR_HANDLING_AVAILABLE else "9001"
                    error_message = "The system experienced an unexpected error. Please try again."
                
                if attempt <= RETRIES:
                    logger.info(f"Retrying in {RETRY_BACKOFF_S * attempt} seconds...")
                    time.sleep(RETRY_BACKOFF_S * attempt)
                else:
                    error_info = _create_error_result(error_code, error_message, {"technical": str(e)})
                    return {"response_text": None, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "error": error_info}

    except Exception as e:
        logger.error(f"An unexpected error occurred in call_gemini_for_extraction: {e}", exc_info=True)
        logger.error(f"Exception type: {type(e)}")
        logger.error(f"Exception args: {e.args}")
        
        # Use the standardized error handler if available
        if ERROR_HANDLING_AVAILABLE:
            try:
                # Use the handle_gemini_exception function to properly classify the error
                sfe = handle_gemini_exception(e, f"Gemini API call for model {model_name}")
                error_info = sfe.to_dict(include_technical=False)
            except Exception as handler_error:
                logger.error(f"Error handler failed: {handler_error}")
                error_code = _classify_llm_error_code(e)
                error_info = _create_error_result(
                    error_code,
                    _get_user_message_for_error(error_code),
                    {"technical": str(e), "type": type(e).__name__, "model": model_name}
                )
        else:
            # Fallback to the old classification logic
            error_code = _classify_llm_error_code(e)
            error_info = _create_error_result(
                error_code,
                _get_user_message_for_error(error_code),
                {"technical": str(e), "type": type(e).__name__, "model": model_name}
            )
        
        return {"response_text": None, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "error": error_info}
    
    return {"response_text": response_text, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "error": None}


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


def _parse_gemini_response_text(response: Any) -> Optional[str]:
    """(kept) safely extracts text content from various Gemini response formats, including truncated responses."""
    try:
        try:
            if hasattr(response, 'text') and response.text:
                return response.text
        except ValueError as e:
            logging.debug(f"response.text access failed: {e}")

        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            finish_reason = getattr(candidate, 'finish_reason', None)
            if finish_reason == 2:
                logging.warning("Response was truncated due to MAX_TOKENS limit")
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


# ------------------------------------------------------------------------------
# JSON parse + REPAIR (kept, with same behavior you asked to maintain)
# ------------------------------------------------------------------------------
def _try_parse_json(logger: logging.Logger, s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception as e:
        _log(logger, logging.DEBUG, f"JSON parse error: {e}")
        return None


def _repair_json(logger: logging.Logger, bad_json: str) -> Optional[Dict[str, Any]]:
    """Keep this logic: fallback LLM call to repair invalid JSON; return dict or None."""
    repair_prompt = (
        "Fix the following into valid JSON. Output only valid JSON, no explanation:\n\n"
        f"```\n{bad_json}\n```"
    )
    try:
        resp = call_gemini_for_extraction(repair_prompt, GEMINI_MODEL_NAME)
        txt = resp.get("response_text") or ""
        return _try_parse_json(logger, txt)
    except Exception as e:
        _log(logger, logging.WARNING, f"JSON repair failed: {e}")
        return None


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main(pdf_path: str, out_dir: Optional[str] = None, verbose: int = 0) -> int:
    logger = build_logger(verbose)
    _log(logger, logging.INFO, "Run started", details={"req_id": getattr(logger, 'req_id', None), "model": GEMINI_MODEL_NAME})

    # 1) optional metadata (kept for filename/context only; not required)
    meta: Dict[str, Any] = {}
    if extract_metadata_dict:
        with timed_step(logger, "extract_metadata"):
            try:
                meta = extract_metadata_dict(pdf_path) or {}
                _log(logger, logging.INFO, "metadata", details=meta)
            except Exception as e:
                _log(logger, logging.WARNING, f"metadata extraction failed: {e}")
    else:
        _log(logger, logging.INFO, "No metadata extractor found (module optional).")

    # 2) read PDF and isolate PART 2 – PRODUCTS
    with timed_step(logger, "text_pipeline"):
        full_text, pdf_error = _read_pdf_text(logger, pdf_path)
        if pdf_error:
            _log(logger, logging.ERROR, "PDF reading failed", 
                 error_code=pdf_error.get("error_code"),
                 details=pdf_error)
            print(json.dumps(pdf_error, indent=2))
            return 2
            
        if not full_text.strip():
            _log(logger, logging.ERROR, "PDF text is empty; cannot proceed.",
                 error_code=ErrorCode.NO_TEXT_CONTENT if ERROR_HANDLING_AVAILABLE else "2003")
            error_result = _create_error_result(
                ErrorCode.NO_TEXT_CONTENT if ERROR_HANDLING_AVAILABLE else "2003",
                "No readable text content was found in the uploaded document.",
                {}
            )
            print(json.dumps(error_result, indent=2))
            return 2
            
        products_text, section_error = _isolate_products(logger, full_text)
        if section_error:
            _log(logger, logging.ERROR, "Section isolation failed",
                 error_code=section_error.get("error_code"),
                 details=section_error)
            print(json.dumps(section_error, indent=2))
            return 3
            
        if not products_text.strip():
            _log(logger, logging.ERROR, "Products section is empty after isolation.",
                 error_code=ErrorCode.NO_PART2_FOUND if ERROR_HANDLING_AVAILABLE else "2004")
            error_result = _create_error_result(
                ErrorCode.NO_PART2_FOUND if ERROR_HANDLING_AVAILABLE else "2004",
                "Could not locate 'PART 2 - PRODUCTS' section in the document.",
                {}
            )
            print(json.dumps(error_result, indent=2))
            return 3

    # 3) build prompt (kept your wording & schema)
    prompt = (
        "You are a world-class construction specification analyst and product extraction system.\n\n"
        "Your task is to extract a structured list of ALL products, materials, or components from the 'PART 2 - PRODUCTS' section of a technical construction specification document. "
        "The input may include page markers, headers, footers, and may be split mid-section or across multiple pages/chunks.\n\n"
        "Follow these steps:\n"
        "1. Locate the section titled 'PART 2 - PRODUCTS'. Extract all content from there until the next major section (e.g., 'PART 3 - EXECUTION') or the end of the text.\n"
        "2. The input may be split across multiple pages or chunks. Extract ALL products, not just the first one.\n"
        "3. For each product/material/component, extract:\n"
        "   - product_name: The specific name or type (e.g., 'Rigid Board Insulation', 'Polyisocyanurate Insulation').\n"
        "   - manufacturers: List all manufacturers explicitly mentioned for that product (empty list if none).\n"
        "   - technical_specifications: A dictionary of key technical properties, standards, and requirements (e.g., compressive strength, R-value, ASTM standards).\n"
        "4. If a product's details are split across pages or chunks, merge them into a single entry.\n"
        "5. Do NOT include information from PART 1 or PART 3, or any unrelated sections.\n"
        "6. Be concise and accurate. Do not hallucinate or invent information. Only use what is present in the text.\n"
        "7. Output a single JSON object with this schema:\n"
        "{\n"
        "  \"products\": [\n"
        "    {\n"
        "      \"product_name\": <string>,\n"
        "      \"manufacturers\": [<string>, ...],\n"
        "      \"technical_specifications\": {<key>: <value>, ...}\n"
        "    }, ...\n"
        "  ]\n"
        "}\n"
        "If no products are found, output {\"products\": []}.\n\n"
        f"Now, extract the products from the following construction specification text (delimited by triple backticks):\n"
        f"```\n{products_text}\n```\n"
        "Output only the JSON object as specified above. Do not include any other text, explanation, or markdown."
    )

    # 4) LLM call with error handling
    with timed_step(logger, "llm_extraction"):
        extraction_result = call_gemini_for_extraction(prompt, GEMINI_MODEL_NAME)
        
        # Check for errors from LLM call
        if extraction_result.get("error"):
            _log(logger, logging.ERROR, "LLM extraction failed",
                 error_code=extraction_result["error"].get("error_code"),
                 details=extraction_result["error"])
            print(json.dumps(extraction_result["error"], indent=2))
            return 4
        
        extracted_json_string = (extraction_result.get("response_text") or "").strip()

    # 5) parse (KEEPING YOUR JSON REPAIR LOGIC)
    parsed_json: Optional[Dict[str, Any]] = None
    if extracted_json_string:
        parsed_json = _try_parse_json(logger, extracted_json_string)
        if parsed_json is None:
            _log(logger, logging.WARNING, "Gemini response was not valid JSON. Attempting repair.")
            parsed_json = _repair_json(logger, extracted_json_string)
    else:
        _log(logger, logging.ERROR, "Failed to get extraction result from LLM (empty text).",
             error_code=ErrorCode.LLM_RESPONSE_INVALID if ERROR_HANDLING_AVAILABLE else "3006")

    # 6) output
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    output_filename = f"{base}_Retrieved.json"
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        output_filepath = os.path.join(out_dir, output_filename)
    else:
        output_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_filename)

    if parsed_json is not None:
        try:
            with open(output_filepath, 'w', encoding="utf-8") as f:
                json.dump(parsed_json, f, ensure_ascii=False, indent=2)
            print(json.dumps(parsed_json, ensure_ascii=False, indent=2))
            _log(logger, logging.INFO, f"Successfully extracted product information and saved to: {output_filepath}")
            return 0
        except IOError as e:
            _log(logger, logging.ERROR, f"Error writing JSON to file {output_filepath}: {e}")
            print(json.dumps(parsed_json, ensure_ascii=False, indent=2))
            return 0
    else:
        error_result = _create_error_result(
            ErrorCode.LLM_RESPONSE_INVALID if ERROR_HANDLING_AVAILABLE else "3006",
            "Unable to extract product details from the uploaded document. Please ensure product specifications are clearly formatted in the PDF or Product Data Sheet.",
            {}
        )
        print(json.dumps(error_result, indent=2))
        return 4


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract PART 2 – PRODUCTS directly from a PDF (no Pinecone).")
    parser.add_argument("pdf_path", help="Path to the input PDF file (e.g., ai/SEC0.pdf)")
    parser.add_argument("--out-dir", help="Directory to write output JSON (default: alongside script)")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v or -vv)")
    args = parser.parse_args()

    sys.exit(main(args.pdf_path, args.out_dir, args.verbose))