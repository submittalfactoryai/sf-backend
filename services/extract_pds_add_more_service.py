from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import json
import os
import re
import logging
from dotenv import load_dotenv
from pathlib import Path
import requests
from requests.exceptions import RequestException
import time
from google.genai.errors import ServerError
import asyncio
import aiohttp
import hashlib
from typing import List, Dict, Any, Tuple, Optional

# NEW: SSL & URL helpers
import ssl
import certifi
from urllib.parse import urljoin, urlparse

# Load environment variables from .env file
dotenv_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
grounded_search_operations_initiated = 0

# COST OPTIMIZATION: Simple in-memory cache for search results
search_cache = {}
CACHE_EXPIRY_HOURS = 24

def get_cache_key(product_data):
    """Create a cache key from product data."""
    product_name = product_data.get('product_name', '')
    manufacturers = product_data.get('manufacturers', [])
    if isinstance(manufacturers, list):
        manufacturers_str = ','.join(sorted(manufacturers))
    else:
        manufacturers_str = str(manufacturers)
    cache_key = f"{product_name}_{manufacturers_str}".lower().replace(' ', '_')
    return cache_key

def is_cache_valid(cache_entry):
    """Check if cache entry is still valid (within expiry time)."""
    if 'timestamp' not in cache_entry:
        return False
    age_hours = (time.time() - cache_entry['timestamp']) / 3600
    return age_hours < CACHE_EXPIRY_HOURS

# Cache management utility functions
def clear_cache():
    """Clear all cache entries."""
    global search_cache
    cache_size = len(search_cache)
    search_cache.clear()
    logging.info(f"Cleared all {cache_size} cache entries")

def clear_cache_entry(product_data):
    """Clear a specific cache entry for given product data."""
    cache_key = get_cache_key(product_data)
    if cache_key in search_cache:
        del search_cache[cache_key]
        logging.info(f"Cleared cache entry for: {cache_key}")
        return True
    return False

def get_cache_stats():
    """Get cache statistics."""
    total_entries = len(search_cache)
    valid_entries = sum(1 for entry in search_cache.values() if is_cache_valid(entry))
    expired_entries = total_entries - valid_entries
    return {
        "total_entries": total_entries,
        "valid_entries": valid_entries,
        "expired_entries": expired_entries,
        "cache_expiry_hours": CACHE_EXPIRY_HOURS
    }

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

client = get_vertex_client()
model_id = "gemini-2.0-flash"

google_search_tool = Tool(google_search=GoogleSearch())

def should_skip_url(url: str) -> bool:
    """Filter out problematic URL patterns known to cause timeouts/failures."""
    if not url:
        return True
    skip_patterns = [
        'google.com/url?q=',
        'redirect.',
        'proxy.',
        '/redirect/',
    ]
    url_lower = url.lower()
    for pattern in skip_patterns:
        if pattern in url_lower:
            return True
    return False

# ===== NEW: Robust verification helpers =====
SSL_CTX = ssl.create_default_context(cafile=certifi.where())

DEFAULT_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

VERTEX_HOST = "vertexaisearch.cloud.google.com"

def _looks_like_pdf_content_type(content_type: str) -> bool:
    return "application/pdf" in (content_type or "").lower()

def _looks_like_pdf_bytes(first_chunk: bytes) -> bool:
    return first_chunk.startswith(b"%PDF-")

_pdf_href_re = re.compile(rb'href=["\']([^"\']+\.pdf)(?:\?[^"\']*)?["\']', re.IGNORECASE)

async def _resolve_vertex_redirect(session: aiohttp.ClientSession, url: str, timeout: aiohttp.ClientTimeout) -> Optional[str]:
    """
    Follow Vertex grounding redirect to the real destination if possible.
    Returns final URL when it leaves the vertex host; otherwise None.
    """
    if VERTEX_HOST not in url:
        return url
    try:
        async with session.get(url, headers=DEFAULT_HEADERS, allow_redirects=True,
                               timeout=timeout, max_redirects=20) as r:
            final_url = str(r.url)
            if VERTEX_HOST in final_url:
                return None
            return final_url
    except Exception as e:
        logging.debug(f"Vertex redirect resolve failed: {e}")
        return None

async def _peek_first_kb(session: aiohttp.ClientSession, url: str, timeout: aiohttp.ClientTimeout) -> Tuple[Optional[str], Optional[bytes], Optional[str]]:
    """
    GET with Range (first 1KB). Return (final_url, first_bytes, content_type).
    """
    headers = dict(DEFAULT_HEADERS)
    headers["Range"] = "bytes=0-1023"
    async with session.get(url, headers=headers, allow_redirects=True,
                           timeout=timeout, max_redirects=20) as resp:
        final_url = str(resp.url)
        ctype = resp.headers.get("Content-Type", "")
        data = await resp.content.read(1024)
        return final_url, data, ctype

async def _extract_pdf_from_html(session: aiohttp.ClientSession, final_url: str, html_snippet: bytes, timeout: aiohttp.ClientTimeout) -> Tuple[bool, Optional[str]]:
    """
    If we got HTML, try to locate a direct .pdf link in the page and verify it.
    We only scan a small snippet (already read). If none found, try to fetch a bit more.
    """
    def try_find(html_bytes: bytes) -> Optional[str]:
        m = _pdf_href_re.search(html_bytes)
        if not m:
            return None
        href = m.group(1).decode("utf-8", errors="ignore")
        # Make absolute
        return urljoin(final_url, href)

    candidate = try_find(html_snippet)
    if not candidate:
        # Fetch up to ~128KB for more chances (without downloading full page)
        headers = dict(DEFAULT_HEADERS)
        headers["Range"] = "bytes=0-131071"
        try:
            async with session.get(final_url, headers=headers, allow_redirects=True,
                                   timeout=timeout, max_redirects=10) as resp:
                more = await resp.content.read(131072)
                candidate = try_find(more)
        except Exception:
            candidate = None

    if not candidate:
        return False, None

    # Verify candidate PDF link quickly (peek bytes)
    try:
        cand_final, first_bytes, cand_ctype = await _peek_first_kb(session, candidate, timeout)
        if _looks_like_pdf_content_type(cand_ctype) or _looks_like_pdf_bytes(first_bytes or b""):
            return True, cand_final
    except Exception:
        pass
    return False, None

async def verify_pdf_link_async(session: aiohttp.ClientSession, pdf_link: str) -> Tuple[bool, Optional[str]]:
    """
    Verify that a link is accessible and is a PDF (or yields a PDF).
    Tries to unwrap Vertex redirects, then peeks first KB for magic bytes.
    Falls back to scraping HTML for direct PDF links.
    """
    try:
        timeout = aiohttp.ClientTimeout(total=20)

        # Unwrap vertex redirect first (if applicable)
        candidate_url = await _resolve_vertex_redirect(session, pdf_link, timeout) or pdf_link

        # Quick HEAD attempt (many servers mislabel or block; failure is OK)
        try:
            async with session.head(candidate_url, headers=DEFAULT_HEADERS,
                                    allow_redirects=True, timeout=timeout) as resp:
                if resp.status == 200 and _looks_like_pdf_content_type(resp.headers.get("Content-Type", "")):
                    return True, str(resp.url)
        except Exception:
            pass  # move on to GET

        # Peek first KB and validate
        try:
            final_url, first_bytes, ctype = await _peek_first_kb(session, candidate_url, timeout)
            if _looks_like_pdf_content_type(ctype) or _looks_like_pdf_bytes(first_bytes or b""):
                return True, final_url

            # If HTML, try to find a .pdf link in page
            if (ctype or "").lower().startswith("text/html"):
                ok, pdf_final = await _extract_pdf_from_html(session, final_url, first_bytes or b"", timeout)
                if ok and pdf_final:
                    return True, pdf_final
        except Exception as e:
            logging.debug(f"Peek/parse failed for {candidate_url}: {e}")

        return False, None

    except Exception as e:
        logging.warning(f"Failed to verify PDF link {pdf_link}: {e}")
        return False, None

async def verify_pdf_links_concurrent(pdf_links: List[str]) -> List[Tuple[bool, Optional[str]]]:
    """
    Verify multiple PDF links concurrently (moderate concurrency to avoid throttling).
    """
    connector = aiohttp.TCPConnector(limit=12, ssl=SSL_CTX)  # NEW: CA bundle & limit
    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [verify_pdf_link_async(session, pdf_link) for pdf_link in pdf_links]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logging.warning(f"Exception verifying {pdf_links[i]}: {result}")
                final_results.append((False, None))
            else:
                final_results.append(result)
        return final_results

def verify_pdf_link(pdf_link):
    """Legacy sync wrapper."""
    async def run_single():
        connector = aiohttp.TCPConnector(limit=1, ssl=SSL_CTX)
        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            is_valid, _ = await verify_pdf_link_async(session, pdf_link)
            return is_valid
    try:
        return asyncio.run(run_single())
    except Exception as e:
        logging.warning(f"Failed to verify PDF link {pdf_link}: {e}")
        return False

def extract_pdf_links(product_data: dict, refresh: bool = False) -> dict:
    """Sync wrapper."""
    return asyncio.run(extract_pdf_links_async(product_data, refresh))

async def extract_pdf_links_async(product_data: dict, refresh: bool = False) -> dict:
    """
    Given product_data (dict), run the Gemini search and return verified PDF links.
    """
    # Check cache
    cache_key = get_cache_key(product_data)
    if not refresh and cache_key in search_cache and is_cache_valid(search_cache[cache_key]):
        logging.info(f"Returning cached results for product: {product_data.get('product_name', 'N/A')}")
        cached_result = search_cache[cache_key]['data'].copy()
        return cached_result

    if refresh and cache_key in search_cache:
        del search_cache[cache_key]
        logging.info(f"Cleared cache for product due to refresh request: {product_data.get('product_name', 'N/A')}")

    final_results = {
        "results": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "model_used": model_id
    }

    print("--------------------------------Calling add more service -------------------------------")

    # State preference from product_data if available
    preferred_state = product_data.get('preferred_state', '')
    state_preference_text = f"\n- STRONGLY PREFER manufacturers and distributors located in {preferred_state}" if preferred_state else ""

    # UPDATED prompt (ask for final .pdf URLs, not redirectors)
    prompt_template = """Find construction product DATA SHEETS for: {product_data}

TASK: Return JSON array with 15-20 PRODUCT DATA SHEET PDF links that subcontractors can use for purchasing decisions.

PRIORITY SEARCH TERMS: "[Product name] product data sheet PDF", "[Product name] technical specifications PDF", "[Product name] cut sheet PDF", "[Product name] spec sheet PDF"

GEOGRAPHIC RESTRICTIONS:
- ONLY search for and return results from United States-based manufacturers, distributors, and suppliers
- BLOCK and EXCLUDE any manufacturers, distributors, or websites from outside the United States
- Do not include results from Canada, Mexico, Europe, Asia, or any other non-US locations{state_preference}

FOCUS ON PROCUREMENT-RELEVANT PDFs:
✅ PRIORITIZE:
- Product data sheets with specifications, dimensions, performance data
- Cut sheets with installation details and ordering information
- Product catalogs with pricing and availability

❌ DEPRIORITIZE/AVOID:
- Safety Data Sheets (SDS/MSDS)
- General company brochures without specific product details
- ASTM/industry standards documents
- Construction specification documents
- Generic marketing materials
- Training materials or manuals

JSON format:
[
  {{
    "heading": "Brief title emphasizing it's a product data sheet",
    "pdf_link": "https://.../final-direct.pdf",
    "pdf_summary": "2-3 sentences focusing on specifications, model numbers, and purchasing relevance",
    "confidence_score": 0.8,
    "justification": "Why this helps subcontractors make purchasing decisions (specs, models, dimensions, etc.)",
    "from_listed_manufacturer": 1
  }}
]

REQUIREMENTS:
- ONLY return JSON array, no other text
- 15-20 results minimum
- Return FINAL DESTINATION URLs that end with .pdf (no redirector/tracking hosts)
- NEVER return links from vertexaisearch.cloud.google.com, google.com/url, or other redirectors
- Focus US construction market ONLY - exclude all non-US sources
- Manufacturer sites preferred for authenticity
- "from_listed_manufacturer": Set to 1 if PDF is from manufacturers listed in product data: {manufacturers}, otherwise 0
- Include model numbers, part numbers, dimensions in summaries when available

RETURN ONLY JSON ARRAY:
"""

    formatted_prompt = prompt_template.format(
        product_data=json.dumps(product_data),
        manufacturers=', '.join(product_data.get('manufacturers', [])),
        state_preference=state_preference_text
    )
    logging.info(f"Starting single comprehensive search for product: {product_data.get('product_name', 'N/A')}")

    max_inner_retries = 1
    inner_retry_delay = 2

    total_prompt_tokens = final_results["prompt_tokens"]
    total_completion_tokens = final_results["completion_tokens"]

    for inner_attempt in range(1, max_inner_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=formatted_prompt,
                config=GenerateContentConfig(
                    temperature=0.2,          # accurate & stable
                    top_p=0.95,
                    candidate_count=1,
                    tools=[google_search_tool],
                    response_modalities=["TEXT"],
                    max_output_tokens=8192
                )
            )

            # Extract and process the AI's response
            if response and hasattr(response, 'candidates') and response.candidates and \
               hasattr(response.candidates[0], 'content') and response.candidates[0].content and \
               hasattr(response.candidates[0].content, 'parts') and response.candidates[0].content.parts:
                ai_response_text_original = response.candidates[0].content.parts[0].text
                logging.info(f"Raw AI response text received:\n---\n{ai_response_text_original}\n---")

                current_prompt_tokens = 0
                current_completion_tokens = 0
                if hasattr(response, 'usage_metadata'):
                    if hasattr(response.usage_metadata, 'prompt_token_count'):
                        current_prompt_tokens = response.usage_metadata.prompt_token_count
                    if hasattr(response.usage_metadata, 'candidates_token_count'):
                        current_completion_tokens = response.usage_metadata.candidates_token_count
                    elif hasattr(response.usage_metadata, 'total_token_count') and current_prompt_tokens > 0:
                        current_completion_tokens = response.usage_metadata.total_token_count - current_prompt_tokens

                    total_prompt_tokens += current_prompt_tokens
                    total_completion_tokens += current_completion_tokens
                    if current_prompt_tokens > 0 or current_completion_tokens > 0:
                        logging.info(f"Token usage for this attempt: Prompt={current_prompt_tokens}, Completion={current_completion_tokens}")

                processed_text = ai_response_text_original.strip()
                if processed_text.startswith('\ufeff'):
                    processed_text = processed_text[1:].strip()

                markdown_match = re.fullmatch(r"```json\s*([\s\S]*?)\s*```", processed_text, re.DOTALL)
                json_to_parse = markdown_match.group(1).strip() if markdown_match else processed_text

                try:
                    llm_results = json.loads(json_to_parse)
                    logging.info(f"Successfully parsed JSON response, found {len(llm_results)} results.")

                    # Filter redirector/tracking links early (still allow Vertex to be resolved during verification if it slipped in)
                    filtered_results = []
                    for result in llm_results:
                        pdf_link = result.get('pdf_link', '')
                        if should_skip_url(pdf_link):
                            logging.info(f"Skipping problematic URL: {pdf_link}")
                            continue
                        filtered_results.append(result)

                    logging.info(f"After filtering, {len(filtered_results)} out of {len(llm_results)} URLs will be verified.")

                    # Verify concurrently
                    pdf_links_to_verify = [result.get('pdf_link', '') for result in filtered_results]
                    verification_results = await verify_pdf_links_concurrent(pdf_links_to_verify)

                    # Build verified results with final URLs
                    verified_results = []
                    for result, (is_valid, final_url) in zip(filtered_results, verification_results):
                        if is_valid and final_url:
                            out = result.copy()
                            if final_url != result.get("pdf_link"):
                                out["vertex_ai_link"] = result.get("pdf_link")  # keep original if it differed
                                out["pdf_link"] = final_url
                            verified_results.append(out)
                            logging.info(f"Verified PDF link: {final_url}")
                        else:
                            logging.warning(f"Invalid or inaccessible PDF link: {result.get('pdf_link', '')}")

                    logging.info(f"After concurrent verification, {len(verified_results)} out of {len(filtered_results)} links are valid.")

                    final_results["prompt_tokens"] = total_prompt_tokens
                    final_results["completion_tokens"] = total_completion_tokens
                    final_results["results"] = verified_results

                    # Cache only if we found verified PDFs
                    if verified_results:
                        search_cache[cache_key] = {
                            'data': final_results.copy(),
                            'timestamp': time.time()
                        }
                        logging.info(f"Cached {len(verified_results)} search results for future use: {cache_key}")
                    else:
                        logging.info(f"Not caching empty results for product: {product_data.get('product_name', 'N/A')}")

                    return final_results

                except json.JSONDecodeError as e:
                    logging.error(f"AI response is not valid JSON: {e}\nRaw (after potential markdown strip):\n---\n{json_to_parse}\n---")
                    if inner_attempt == max_inner_retries:
                        break
            else:
                logging.error("No valid response, candidates, content, or parts found in the AI model response to parse JSON.")
                if inner_attempt == max_inner_retries:
                    break

        except (ServerError) as e:
            error_msg = str(e)
            logging.warning(f"Inner attempt {inner_attempt}/{max_inner_retries} failed: {error_msg}")
            if inner_attempt == max_inner_retries:
                break
            sleep_time = inner_retry_delay * (2 ** (inner_attempt - 1))
            logging.info(f"Retrying inner attempt in {sleep_time} seconds...")
            await asyncio.sleep(sleep_time)

        except Exception as e:
            logging.error(f"Unexpected error during PDF link extraction: {str(e)}")
            if inner_attempt == max_inner_retries:
                break
            sleep_time = inner_retry_delay * (2 ** (inner_attempt - 1))
            logging.info(f"Retrying inner attempt in {sleep_time} seconds...")
            await asyncio.sleep(sleep_time)

    # All retries failed
    final_results["prompt_tokens"] = total_prompt_tokens
    final_results["completion_tokens"] = total_completion_tokens
    final_results["results"] = []
    return final_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract PDF links for a product using Gemini")
    parser.add_argument("--product_json", type=str, required=True, help="Path to a JSON file containing product data")
    args = parser.parse_args()
    with open(args.product_json, "r") as f:
        product_data = json.load(f)
    results_data = extract_pdf_links(product_data)
    print(json.dumps(results_data, indent=2))
