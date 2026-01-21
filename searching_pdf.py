import os
# Keep the correct GoogleSearch import
from serpapi import GoogleSearch
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
import requests
import json
import traceback
import logging

# Configure logger for this module
logger = logging.getLogger(__name__)

# Load environment variables from the parent directory's .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not SERPAPI_API_KEY:
    logger.critical("SERPAPI_API_KEY not found in environment variables. PDF search will fail.")
    # Consider raising an exception here if the key is absolutely required for the app to function
    # raise ValueError("SERPAPI_API_KEY is not set in the environment.")

def search_pdfs_serpapi(
    product_name: str,
    state_name: Optional[str] = None,
    manufacturers: Optional[list] = None,
    num_results: int = 15,
    custom_query: Optional[str] = None
) -> tuple[List[Dict[str, str]], str]:
    """
    Searches Google using SerpApi for PDF data sheets related to a product,
    optionally filtered by state and manufacturers, focusing on relevance and speed.

    Args:
        product_name: The name of the product.
        state_name: An optional US state name to include in the search query.
        manufacturers: Optional list of manufacturer names to include in the query.
        num_results: The desired number of search results.
        custom_query: If provided, use this as the search query directly.

    Returns:
        A tuple (results, query_used):
            results: list of dicts with 'title', 'link', and 'snippet'.
            query_used: the actual query string sent to SerpApi.
    """
    if not SERPAPI_API_KEY:
        logger.error("SerpApi API key is not configured. Cannot perform search.")
        return [], ""

    # Build the query string
    if custom_query:
        query = custom_query
    else:
        query_parts = [
            f'{product_name}',
            '"Product Data Sheet Pdf"'
        ]
        if state_name:
            query_parts.append(state_name)
        if manufacturers:
            query_parts.extend(manufacturers)
        query = " ".join(query_parts)

    logger.info(f"Executing SerpApi Query: {query}")

    params = {
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": num_results,
        "engine": "google",
        "gl": "us",
        "hl": "en",
        "google_domain": "google.com",
    }

    logger.debug(f"SerpApi Params: {params}")

    try:
        logger.info("Attempting SerpApi search...")
        # Use GoogleSearch imported from serpapi
        search = GoogleSearch(params)
        logger.info("SerpApi search object created. Getting dictionary...")
        results_data = search.get_dict()
        logger.info("SerpApi dictionary received. Processing results...")
        # logger.debug(f"Raw SerpApi results: {json.dumps(results_data, indent=2)}")  # Keep commented unless debugging

        processed_results: List[Dict[str, str]] = []
        processed_links = set()

        if "organic_results" in results_data:
            for result in results_data["organic_results"]:
                link = result.get("link", "")
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                # Only include links that end with .pdf
                if link and link.lower().endswith('.pdf') and link not in processed_links:
                    processed_results.append({
                        "title": title if title else "Untitled PDF",
                        "link": link,
                        "snippet": snippet if snippet else "No description available."
                    })
                    processed_links.add(link)

        if "inline_pdfs" in results_data:
            for result in results_data["inline_pdfs"]:
                link = result.get("link", "")
                # Only include links that end with .pdf
                if link and link.lower().endswith('.pdf') and link not in processed_links:
                    processed_results.append({
                        "title": result.get("title", "PDF Document"),
                        "link": link,
                        "snippet": result.get("snippet", "Inline PDF result.")
                    })
                    processed_links.add(link)

        if not processed_results:
            logger.info("No relevant PDF results found in SerpApi response.")
        else:
            logger.info(f"Processed {len(processed_results)} potential PDF results from SerpApi.")

        logger.info("Finished processing SerpApi results.")
        return processed_results, query

    except requests.exceptions.Timeout:
        logger.warning("SerpApi request timed out. Check network or API status.")
    except requests.exceptions.ConnectionError:
        logger.warning("Network connection error during SerpApi request. Check connectivity.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during SerpApi search: {e}", exc_info=True)
        # Check if the error message indicates an API key issue or other SerpApi specific problem
        if "API key" in str(e):
             logger.error("SerpApi Error Detail: Check your SERPAPI_API_KEY.")

    return [], query  # Return empty list and query on error
