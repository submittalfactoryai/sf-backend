from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import json
import os  # Added for environment variable access
import re  # Moved to top
import logging  # Added for logging
from dotenv import load_dotenv  # Add this import
from pathlib import Path # Added import
import requests  # Added for HTTP requests
from requests.exceptions import RequestException  # Added for exception handling
import time  # Added for rate limiting
from google.genai.errors import ServerError  # Fixed import - only include classes that exist
import asyncio
import aiohttp
import hashlib
from typing import List, Dict, Any, Tuple, Optional

# Load environment variables from .env file
# Ensure .env is in the same directory as this script (backend/)
dotenv_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
grounded_search_operations_initiated = 0

# COST OPTIMIZATION: Simple in-memory cache for search results
# Cache key: hash of product name + manufacturers
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

# Replace with your API key, preferably from an environment variable
# Example: api_key = os.getenv("GEMINI_API_KEY")
# Ensure you set this environment variable in your system.
api_key = os.getenv("GEMINI_API_KEY")  # Load environment variable for API key
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Please ensure it is set in backend/.env and the file is being loaded correctly.")

client = genai.Client(api_key=api_key)
model_id = "gemini-2.0-flash"

google_search_tool = Tool(
    google_search=GoogleSearch()
)

def should_skip_url(url):
    """
    Filter out problematic URL patterns that are known to cause timeouts or failures.
    
    Args:
        url (str): The URL to check
        
    Returns:
        bool: True if the URL should be skipped, False if it should be verified
    """
    if not url:
        return True
        
    skip_patterns = [
        # Removed 'grounding-api-redirect' and 'vertexaisearch.cloud.google.com' 
        # as these are legitimate URLs provided by the Gemini API
        'google.com/url?q=',
        'redirect.',
        'proxy.',
        '/redirect/',
        # Add other problematic patterns as they're discovered
    ]
    
    url_lower = url.lower()
    for pattern in skip_patterns:
        if pattern in url_lower:
            return True
    
    return False

async def verify_pdf_link_async(session: aiohttp.ClientSession, pdf_link: str) -> Tuple[bool, Optional[str]]:
    """
    Async version: Verify that a link is accessible and points to a PDF file.
    
    Args:
        session: aiohttp ClientSession
        pdf_link (str): The URL to verify
        
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, original_url_if_redirected)
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Make a HEAD request first to check the content type without downloading the entire file
        timeout = aiohttp.ClientTimeout(total=10)
        
        async with session.head(pdf_link, headers=headers, timeout=timeout, allow_redirects=True) as response:
            final_url = str(response.url)  # This gives us the final URL after redirects
            
            # If HEAD request fails, try a GET request 
            if response.status != 200:
                async with session.get(pdf_link, headers=headers, timeout=timeout, allow_redirects=True) as get_response:
                    final_url = str(get_response.url)
                    response = get_response
            
            # Check if the response is successful and the content type is PDF
            content_type = response.headers.get('Content-Type', '').lower()
            is_valid = (response.status == 200 and 
                       ('application/pdf' in content_type or 
                        pdf_link.lower().endswith('.pdf')))  # Fallback check if content-type is unreliable
            
            return is_valid, final_url if is_valid else None
                
    except Exception as e:
        logging.warning(f"Failed to verify PDF link {pdf_link}: {str(e)}")
        return False, None

async def verify_pdf_links_concurrent(pdf_links: List[str]) -> List[Tuple[bool, Optional[str]]]:
    """
    Verify multiple PDF links concurrently for blazing fast verification.
    
    Args:
        pdf_links: List of PDF URLs to verify
        
    Returns:
        List of tuples: [(is_valid, original_url_if_valid), ...]
    """
    connector = aiohttp.TCPConnector(limit=20)  # Limit concurrent connections
    timeout = aiohttp.ClientTimeout(total=10)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [verify_pdf_link_async(session, pdf_link) for pdf_link in pdf_links]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logging.warning(f"Exception verifying {pdf_links[i]}: {result}")
                final_results.append((False, None))
            else:
                final_results.append(result)
        
        return final_results

def verify_pdf_link(pdf_link):
    """
    Legacy sync function - now just calls the async version
    """
    async def run_single():
        connector = aiohttp.TCPConnector(limit=1)
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            is_valid, _ = await verify_pdf_link_async(session, pdf_link)
            return is_valid
    
    try:
        return asyncio.run(run_single())
    except Exception as e:
        logging.warning(f"Failed to verify PDF link {pdf_link}: {str(e)}")
        return False

def extract_pdf_links(product_data: dict, refresh: bool = False) -> dict:
    """
    Given product_data (dict), run the Gemini search and return a list of PDF link results.
    With error handling and retries for API failures.
    Now also returns token usage and model info.
    
    Args:
        product_data (dict): Product data to search for
        refresh (bool): If True, bypass cache and force new search
    """
    return asyncio.run(extract_pdf_links_async(product_data, refresh))

async def extract_pdf_links_async(product_data: dict, refresh: bool = False) -> dict:
    """
    Given product_data (dict), run the Gemini search and return a list of PDF link results.
    With error handling and retries for API failures.
    Now also returns token usage and model info.
    
    Args:
        product_data (dict): Product data to search for
        refresh (bool): If True, bypass cache and force new search
    """
    # COST OPTIMIZATION: Check cache first (unless refresh is requested)
    cache_key = get_cache_key(product_data)
    if not refresh and cache_key in search_cache and is_cache_valid(search_cache[cache_key]):
        logging.info(f"Returning cached results for product: {product_data.get('product_name', 'N/A')}")
        cached_result = search_cache[cache_key]['data'].copy()
        return cached_result
    
    # If refresh is requested, clear this cache entry
    if refresh and cache_key in search_cache:
        del search_cache[cache_key]
        logging.info(f"Cleared cache for product due to refresh request: {product_data.get('product_name', 'N/A')}")
    
    final_results = {
        "results": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "model_used": model_id # Use the globally defined model_id
    }
    
    # COST OPTIMIZATION: Remove expensive outer retry loop entirely
    # Focus on getting comprehensive results in single high-quality search
    
    # Extract state preference from product_data if available
    preferred_state = product_data.get('preferred_state', '')
    state_preference_text = ""
    if preferred_state:
        state_preference_text = f"\n- STRONGLY PREFER manufacturers and distributors located in {preferred_state}"
    
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
- Safety Data Sheets (SDS/MSDS) - these don't help with purchasing
- General company brochures without specific product details
- ASTM/industry standards documents
- Installation guides with product specifications
- Construction specification documents
- Generic marketing materials
- Training materials or manuals

JSON format:
[
  {{
    "heading": "Brief title emphasizing it's a product data sheet",
    "pdf_link": "direct_url.pdf", 
    "pdf_summary": "2-3 sentences focusing on specifications, model numbers, and purchasing relevance",
    "confidence_score": 0.8,
    "justification": "Why this helps subcontractors make purchasing decisions (specs, models, dimensions, etc.)",
    "from_listed_manufacturer": 1
  }}
]

REQUIREMENTS:
- ONLY return JSON array, no other text
- 15-20 results minimum
- Direct URLs ending in .pdf only
- Focus US construction market ONLY - exclude all non-US sources
- Manufacturer sites preferred for authenticity
- "from_listed_manufacturer": Set to 1 if PDF is from manufacturers listed in product data: {manufacturers}, otherwise 0
- Prioritize PDFs from listed manufacturers
- EMPHASIZE product data sheets over safety sheets, brochures, or standards
- Include model numbers, part numbers, dimensions in summaries when available

RETURN ONLY JSON ARRAY:"""
    
    formatted_prompt = prompt_template.format(
        product_data=json.dumps(product_data),
        manufacturers=', '.join(product_data.get('manufacturers', [])),
        state_preference=state_preference_text
    )
    logging.info(f"Starting single comprehensive search for product: {product_data.get('product_name', 'N/A')}")
    
    # COST OPTIMIZATION: Reduce inner retry from 3 to 1 attempt for network errors only
    max_inner_retries = 1
    inner_retry_delay = 2
    
    # Track total tokens
    total_prompt_tokens = final_results["prompt_tokens"]
    total_completion_tokens = final_results["completion_tokens"]
    
    # Single retry loop for network errors only
    for inner_attempt in range(1, max_inner_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=formatted_prompt,
                config=GenerateContentConfig(
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

                # Attempt to get token counts from usage_metadata
                current_prompt_tokens = 0
                current_completion_tokens = 0
                if hasattr(response, 'usage_metadata'):
                    if hasattr(response.usage_metadata, 'prompt_token_count'):
                        current_prompt_tokens = response.usage_metadata.prompt_token_count
                    if hasattr(response.usage_metadata, 'candidates_token_count'): # or 'completion_token_count'
                        current_completion_tokens = response.usage_metadata.candidates_token_count
                    elif hasattr(response.usage_metadata, 'total_token_count') and current_prompt_tokens > 0:
                        # If only total is available, estimate completion if prompt is known
                        current_completion_tokens = response.usage_metadata.total_token_count - current_prompt_tokens
                    
                    # Add current token usage to total
                    total_prompt_tokens += current_prompt_tokens
                    total_completion_tokens += current_completion_tokens
                    if current_prompt_tokens > 0 or current_completion_tokens > 0:
                         logging.info(f"Token usage for this attempt: Prompt={current_prompt_tokens}, Completion={current_completion_tokens}")

                processed_text = ai_response_text_original.strip()
                if processed_text.startswith('\ufeff'):
                    processed_text = processed_text[1:]
                    processed_text = processed_text.strip()
                markdown_match = re.fullmatch(r"```json\s*([\s\S]*?)\s*```", processed_text, re.DOTALL)
                if markdown_match:
                    content_from_markdown = markdown_match.group(1)
                    cleaned_markdown_content = content_from_markdown.strip()
                    if cleaned_markdown_content.startswith('\ufeff'):
                        cleaned_markdown_content = cleaned_markdown_content[1:]
                        cleaned_markdown_content = cleaned_markdown_content.strip()
                    json_to_parse = cleaned_markdown_content
                else:
                    json_to_parse = processed_text
                try:
                    llm_results = json.loads(json_to_parse)
                    logging.info(f"Successfully parsed JSON response, found {len(llm_results)} results.")
                    
                    # COST OPTIMIZATION: Filter out problematic URLs before expensive verification
                    filtered_results = []
                    for result in llm_results:
                        pdf_link = result.get('pdf_link', '')
                        if should_skip_url(pdf_link):
                            logging.info(f"Skipping problematic URL: {pdf_link}")
                            continue
                        filtered_results.append(result)
                    
                    logging.info(f"After filtering, {len(filtered_results)} out of {len(llm_results)} URLs will be verified.")
                    
                    # BLAZING FAST: Verify PDF links concurrently instead of sequentially
                    pdf_links_to_verify = [result.get('pdf_link', '') for result in filtered_results]
                    
                    # Run concurrent verification
                    verification_results = await verify_pdf_links_concurrent(pdf_links_to_verify)
                    
                    # Build verified results with original URLs
                    verified_results = []
                    for i, (result, (is_valid, original_url)) in enumerate(zip(filtered_results, verification_results)):
                        if is_valid and original_url:
                            # Update result with original URL while keeping vertex AI URL for reference
                            result_copy = result.copy()
                            result_copy['vertex_ai_link'] = result_copy['pdf_link']  # Store original vertex AI link
                            result_copy['pdf_link'] = original_url  # Replace with original website URL
                            verified_results.append(result_copy)
                            logging.info(f"Verified PDF link: {original_url} (from {result.get('pdf_link', '')})")
                        else:
                            logging.warning(f"Invalid or inaccessible PDF link: {result.get('pdf_link', '')}")
                    
                    logging.info(f"After concurrent verification, {len(verified_results)} out of {len(filtered_results)} links are valid.")
                    
                    # Update the total tokens in the final results
                    final_results["prompt_tokens"] = total_prompt_tokens
                    final_results["completion_tokens"] = total_completion_tokens
                    final_results["results"] = verified_results
                    
                    # COST OPTIMIZATION: Only cache results if we found verified PDFs
                    # Don't cache empty results so refresh can try again
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
                    # If we're on the last inner retry, return empty results
                    if inner_attempt == max_inner_retries:
                        break

            else:
                logging.error("No valid response, candidates, content, or parts found in the AI model response to parse JSON.")
                # If we're on the last inner retry, return empty results
                if inner_attempt == max_inner_retries:
                    break

        except (ServerError) as e:
            error_msg = str(e)
            logging.warning(f"Inner attempt {inner_attempt}/{max_inner_retries} failed: {error_msg}")
            
            # If this is the last inner attempt, return empty results
            if inner_attempt == max_inner_retries:
                break
            
            # Exponential backoff for inner retry
            sleep_time = inner_retry_delay * (2 ** (inner_attempt - 1))
            logging.info(f"Retrying inner attempt in {sleep_time} seconds...")
            await asyncio.sleep(sleep_time)
            
        except Exception as e:
            logging.error(f"Unexpected error during PDF link extraction: {str(e)}")
            # If this is the last inner attempt, return empty results
            if inner_attempt == max_inner_retries:
                break
            
            # Add some delay before inner retry
            sleep_time = inner_retry_delay * (2 ** (inner_attempt - 1))
            logging.info(f"Retrying inner attempt in {sleep_time} seconds...")
            await asyncio.sleep(sleep_time)
    
    # If we reach here, all retries failed
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
