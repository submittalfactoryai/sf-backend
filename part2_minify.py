# part2_minify.py
"""
PART 2 Minification Module with Standardized Error Handling
Condenses PART 2 - PRODUCTS text for efficient LLM processing.

Updated with SOW Section 6 compliant error handling.
"""

import os
import re
import time
import logging
import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import standardized error handling
try:
    from error_handlers import (
        ErrorCode,
        classify_llm_error,
        SubmittalFactoryException,
        handle_gemini_exception,
    )
    ERROR_HANDLING_AVAILABLE = True
except ImportError:
    try:
        from error_handlers import (
            ErrorCode,
            classify_llm_error,
            SubmittalFactoryException,
            handle_gemini_exception,
        )
        ERROR_HANDLING_AVAILABLE = True
    except ImportError:
        ERROR_HANDLING_AVAILABLE = False
        class ErrorCode:
            MODEL_NOT_FOUND = "3001"
            API_KEY_INVALID = "3003"
            RATE_LIMITED = "3004"
            LLM_TIMEOUT = "3007"
            LLM_RESPONSE_INVALID = "3006"
            MODEL_OVERLOADED = "3002"
            INTERNAL_ERROR = "9001"

# ---- Config ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Model & limits
MODEL_MINIFY = os.getenv("GEMINI_MINIFY_MODEL", "gemini-2.5-flash")
DEFAULT_TARGET_CHARS = int(os.getenv("PART2_MINIFY_TARGET_CHARS", "25000"))
HARD_MAX_CHARS = int(os.getenv("PART2_MINIFY_HARD_MAX_CHARS", "30000"))

# Vertex AI Configuration
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "submittalfactoryai")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

# Set credentials path for Google Cloud SDK
if GOOGLE_APPLICATION_CREDENTIALS:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

# Initialize Vertex AI client
def get_vertex_client():
    """Get or create Vertex AI client."""
    return genai.Client(
        vertexai=True,
        project=GOOGLE_CLOUD_PROJECT,
        location=GOOGLE_CLOUD_LOCATION
    )

# File exports
PROMPT_EXPORT_DIR = os.getenv("PROMPT_EXPORT_DIR", "./prompts")
ARTIFACT_EXPORT_DIR = os.getenv("ARTIFACT_EXPORT_DIR", "./artifacts")

MINIFY_PROMPT_TEMPLATE = """
You are a *construction specification condenser*.

Goal: Reduce the following **PART 2 – PRODUCTS** text to <= {target_chars} characters (hard max {hard_max_chars}), while **preserving every product line and all context strictly needed** for downstream product extraction.

### Keep (must preserve)
- Section numbering and headings inside PART 2 (e.g., "2.1", "2.2", "2.3", and named sub-articles).
- Product/Material headers and named items.
- All specification bullets that define product identity: standards (ASTM/UL/EN/etc.), type/grade/class, dimensions, performance values, composition, finishes, accessories that are *orderable*, and *explicit* options/alternates.
- "Basis of Design" lines (as textual references), **but do not expand narrative copies**.
- Manufacturer lists (approved/base/optional).
- Cross-references within PART 2 that constrain selection (e.g., "match finish in 2.5.B").
- Any explicit selection tables or enumerated options that affect the product identity.
- Submittal phrases **only** when they define required certification/standard marking tied to product identity (e.g., "Provide ICC-ES report for Type X boards").

### Drop / Compress
- Boilerplate, general notes, execution language, QA narratives unless they define a product identity constraint.
- Repetitive synonyms, restated paragraphs, marketing fluff, and long prose that does not add unique spec keys.
- Page headers/footers, line numbers, orphan page artifacts.

### Style
- Output **plain text only** (no JSON, no markdown fences), preserving logical hierarchy:
  - Keep section numbers and short headers.
  - Convert multi-line rambles into clean bullets or short lines.
  - Remove excessive whitespace; keep single blank line between major subsections.

### Strict Requirements
- Do **not** hallucinate or invent anything not in the input.
- Do **not** re-label standards or change values.
- Ensure the result stays **<= {hard_max_chars} characters**. If needed, progressively compress commentary but never drop product-identity bullets.
- Return **only** the condensed plain text. No explanations.

--- BEGIN PART 2 TEXT ---
{part2_text}
--- END PART 2 TEXT ---
"""


def _create_error_result(error_code: str, message: str, details: Optional[Dict] = None) -> Dict[str, Any]:
    """Create a standardized error result dictionary."""
    return {
        "success": False,
        "error_code": error_code,
        "error_message": message,
        "details": details or {}
    }


def _classify_minify_error(error: Exception) -> Tuple[str, str]:
    """
    Classify minification errors and return (error_code, user_message).
    """
    error_str = str(error).lower()
    
    if "404" in error_str or "not found" in error_str:
        return (
            ErrorCode.MODEL_NOT_FOUND if ERROR_HANDLING_AVAILABLE else "3001",
            "The AI extraction service is temporarily unavailable. Our team has been notified."
        )
    elif "timeout" in error_str or "deadline" in error_str or "504" in error_str:
        return (
            ErrorCode.LLM_TIMEOUT if ERROR_HANDLING_AVAILABLE else "3007",
            "The AI processing took too long. Please try with a smaller document."
        )
    elif "429" in error_str or "rate limit" in error_str or "quota" in error_str:
        return (
            ErrorCode.RATE_LIMITED if ERROR_HANDLING_AVAILABLE else "3004",
            "You've reached the temporary request limit. Please wait before trying again."
        )
    elif "401" in error_str or "403" in error_str or "unauthorized" in error_str:
        return (
            ErrorCode.API_KEY_INVALID if ERROR_HANDLING_AVAILABLE else "3003",
            "AI service authentication failed. Please contact support."
        )
    elif "503" in error_str or "overloaded" in error_str or "capacity" in error_str:
        return (
            ErrorCode.MODEL_OVERLOADED if ERROR_HANDLING_AVAILABLE else "3002",
            "The AI service is experiencing high demand. Please wait a moment and try again."
        )
    else:
        return (
            ErrorCode.INTERNAL_ERROR if ERROR_HANDLING_AVAILABLE else "9001",
            "The system experienced an unexpected error. Please try again."
        )


def _timestamped_name(base: str, suffix: str) -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_base = re.sub(r"[^A-Za-z0-9._\\-]+", "_", base)
    return f"{safe_base}__{ts}.{suffix}"


def _export_text(content: str, directory: str, suggested_base: str, suffix: str = "txt") -> str:
    out_dir = Path(directory).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = _timestamped_name(Path(suggested_base).stem, suffix)
    fpath = out_dir / fname
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)
    logging.info(f"Exported to: {fpath}")
    return str(fpath)


def _build_minify_prompt(part2_text: str, target_chars: int, hard_max_chars: int) -> str:
    return MINIFY_PROMPT_TEMPLATE.format(
        target_chars=target_chars,
        hard_max_chars=hard_max_chars,
        part2_text=part2_text
    )


def _choose_api_key() -> Optional[str]:
    """
    Prefer GOOGLE_API_KEY; fall back to GEMINI_API_KEY if set.
    """
    g = os.getenv("GOOGLE_API_KEY")
    alt = os.getenv("GEMINI_API_KEY")
    if g and alt:
        logging.warning("Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.")
        return g
    return g or alt


def _gemini_minify_call(prompt: str, model_name: str) -> Tuple[str, Optional[Dict]]:
    """
    Direct call to Gemini via Vertex AI; returns (plain_text, error_dict).
    If successful, error_dict is None.
    """
    try:
        client = get_vertex_client()
    except Exception as e:
        logging.error(f"Failed to initialize Vertex AI client: {e}")
        return "", _create_error_result(
            ErrorCode.API_KEY_INVALID if ERROR_HANDLING_AVAILABLE else "3003",
            "AI service authentication failed. Please contact support.",
            {"technical": f"Vertex AI client init failed: {e}"}
        )

    generation_config = types.GenerateContentConfig(
        max_output_tokens=65000,   # Vertex AI max is 65536
        temperature=0.1,
    )

    logging.info(f"Minify Sending request to Gemini model: {model_name} via Vertex AI (for part2 minification)")
    logging.info(f"Before Minify Prompt length: {len(prompt)} characters")

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=generation_config,
        )
    except Exception as e:
        error_code, user_message = _classify_minify_error(e)
        es = str(e).lower()
        if "deadline exceeded" in es or "timeout" in es or "504" in es:
            logging.error(f"Gemini timeout during minification: {e}")
        else:
            logging.error(f"Gemini error during minification: {e}", exc_info=True)
        return "", _create_error_result(error_code, user_message, {"technical": str(e)})

    # Try to pull text robustly
    txt = ""
    try:
        if getattr(response, "text", None):
            txt = response.text or ""
        elif getattr(response, "candidates", None):
            cand = response.candidates[0]
            if getattr(cand, "content", None) and getattr(cand.content, "parts", None):
                txt = "".join(getattr(p, "text", "") or "" for p in cand.content.parts)
    except Exception:
        logging.warning("Failed to read response.text; attempting to read parts.", exc_info=True)

    # Optional: log usage
    try:
        if getattr(response, "usage_metadata", None):
            logging.info(
                f"Minify Gemini Usage - Prompt: "
                f"{response.usage_metadata.prompt_token_count}, "
                f"Completion: {response.usage_metadata.candidates_token_count}"
            )
    except Exception:
        pass

    if not txt.strip():
        return "", _create_error_result(
            ErrorCode.LLM_RESPONSE_INVALID if ERROR_HANDLING_AVAILABLE else "3006",
            "Failed to get a valid response from the AI service.",
            {"response_type": type(response).__name__}
        )

    return txt.strip(), None


def _soft_rule_minify(text: str, target_chars: int, hard_max_chars: int) -> str:
    """
    Deterministic fallback minifier:
    - Keeps section headers (2.x), bullets, manufacturer lines, and spec-like lines (ASTM/UL/ISO/Type/Grade/etc.)
    - Trims boilerplate and collapses whitespace
    """
    lines = text.splitlines()
    keep = []
    spec_regex = re.compile(r"\b(ASTM|EN|UL|FM|ISO|DIN|ANSI|ICC|IES|Type|Grade|Class|PSI|MPa|kPa|in\.?|mm|%|R[- ]?value|UL\s?Listed|NFPA)\b", re.I)
    sec_head = re.compile(r"^\s*2\.\d+(\.|[A-Za-z])?")
    bullet = re.compile(r"^\s*([\-–•*]|[a-zA-Z]\.)\s+")
    manufacturer_line = re.compile(r"\b(Manufacturers|Approved|Basis of Design|Acceptable|Substitutions)\b", re.I)

    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        # Drop obvious page junk
        if re.search(r"^\s*(Page|SECTION|Project Name|Project No\.)\b", s, re.I):
            continue
        if sec_head.search(s) or bullet.search(s) or spec_regex.search(s) or manufacturer_line.search(s):
            keep.append(s)
        else:
            # short product-ish lines
            if len(s) <= 140 and any(k in s.lower() for k in [
                "insulation","gypsum","panel","fastener","sealant","adhesive","membrane","board",
                "steel","aluminum","glass","tile","paint","door","hardware","pipe","valve"
            ]):
                keep.append(s)

    condensed = "\n".join(keep)
    condensed = re.sub(r"\n{3,}", "\n\n", condensed)

    if len(condensed) > hard_max_chars:
        condensed = condensed[:hard_max_chars]
    return condensed


def minify_part2_with_llm(
    part2_text: str,
    export_prefix: Optional[str] = None,
    target_chars: Optional[int] = None,
    hard_max_chars: Optional[int] = None,
    model_name: Optional[str] = None,
    max_retries: int = 2
) -> Tuple[str, bool]:
    """
    Minify PART 2 via a direct Gemini call with a safe fallback.
    Now includes standardized error handling.
    
    Returns: (minified_text, used_fallback)
    """
    logger = logging.getLogger(__name__)

    target = target_chars or DEFAULT_TARGET_CHARS
    hardmax = hard_max_chars or HARD_MAX_CHARS
    model = model_name or MODEL_MINIFY

    # 1) Try LLM minification with limited retries
    for attempt in range(1, max_retries + 1):
        try:
            prompt = _build_minify_prompt(part2_text, target, hardmax)

            txt, error = _gemini_minify_call(prompt, model)

            if error:
                logger.warning(f"Minify attempt {attempt}: LLM error - {error.get('error_message', 'Unknown')}")
                time.sleep(1.2 * attempt)
                continue

            if not txt:
                logger.warning(f"Minify attempt {attempt}: empty response.")
                time.sleep(1.2 * attempt)
                continue

            if len(txt) > hardmax:
                logging.info(f"Minified exceeds hard cap ({len(txt)} > {hardmax}); truncating.")
                txt = txt[:hardmax]

            # quick sanity: should contain spec signals or section markers
            if not re.search(r"\b2\.\d", txt) and not re.search(r"\b(ASTM|UL|EN|ISO|Type|Grade|Class)\b", txt, re.I):
                logger.warning("Minified text failed sanity check; retrying…")
                time.sleep(1.2 * attempt)
                continue

            logging.info(f"Minified PART 2 length: {len(txt)} chars (llm)")
            return txt, False
            
        except Exception as e:
            logger.error(f"Minify attempt {attempt} failed: {e}", exc_info=True)
            time.sleep(1.2 * attempt)

    # 2) Fallback to rule-based
    logger.warning("LLM minify failed; using rule-based fallback.")
    fallback = _soft_rule_minify(part2_text, target, hardmax)
    try:
        _export_text(fallback, ARTIFACT_EXPORT_DIR, (export_prefix or "PART2_minified_fallback"))
    except Exception as exp_err:
        logger.warning(f"Failed to export fallback minified artifact: {exp_err}")
    logging.info(f"Minified PART 2 length: {len(fallback)} chars (fallback)")
    return fallback, True