import sys
import os
import json
import re
import logging
import concurrent.futures
import traceback
import fitz  # Added for PyMuPDF
from google import genai
from google.genai import types
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MODEL_NAME = "gemini-1.5-flash" # Use a fast model for metadata
TIMEOUT_SECONDS = 60

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

# --- Helper Functions ---

def extract_first_page_text(pdf_path: str) -> Optional[str]:
    """Extracts text from the first page of a PDF using PyMuPDF (fitz)."""
    try:
        with fitz.open(pdf_path) as doc:
            if doc.page_count > 0:
                first_page = doc.load_page(0) # Load the first page (index 0)
                text = first_page.get_text("text")
                return text
            else:
                logging.warning(f"PDF has no pages: {pdf_path}")
                return None
    except Exception as e:
        logging.error(f"Error reading PDF {pdf_path}: {e}")
        traceback.print_exc()
        return None

def _parse_gemini_response_text(response: Any) -> Optional[str]:
    """Safely extracts text content from various Gemini response formats."""
    try:
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'parts') and response.parts:
            return "".join(part.text for part in response.parts if hasattr(part, 'text'))
        else:
            logging.warning(f"Unexpected Gemini API response format: {type(response)}")
            return None
    except Exception as e:
        logging.error(f"Error processing Gemini response parts: {e}")
        try:
            logging.debug(f"Raw Gemini response on error: {response}")
        except Exception:
            logging.debug("Could not log raw Gemini response.")
        return None


def _extract_json_from_text(text: str) -> Optional[str]:
    """Extracts JSON content, potentially embedded in markdown code blocks."""
    match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```|\{.*\}', text, re.DOTALL)
    if match:
        json_content = match.group(1) if match.group(1) else match.group(0)
        return json_content.strip()
    else:
        logging.warning(f"Response does not appear to be JSON or wrapped in markdown: {text[:100]}...")
        return None


def call_gemini_for_metadata(text_content: str) -> Dict[str, Any]:
    """Calls Gemini to extract metadata, returns dict with data and token counts."""
    metadata_dict = {}
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    if not text_content:
        logging.warning("No text content provided for metadata extraction.")
        return {
            "data": metadata_dict,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }

    prompt = f"""
    Analyze the following text from the first page of a construction specification document. Extract the following metadata fields:
    - project_name: The official name of the project.
    - project_number: The identifier or number assigned to the project.
    - spec_section_number: The specification section number (e.g., "09 65 19").
    - spec_section_title: The title of the specification section (e.g., "Resilient Tile Flooring").
    - publication_date: The date the specification document was published or issued.

    Return the result ONLY as a single, valid JSON object with these exact keys. If a field cannot be found, use a JSON null value for that key. Do not include explanations or markdown.

    Text:
    ---
    {text_content[:4000]}
    ---

    JSON Output:
    """

    try:
        client = get_vertex_client()
        
        generation_config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=1024,
        )

        logging.info(f"Sending metadata request to Gemini model: {MODEL_NAME} via Vertex AI")

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                client.models.generate_content,
                model=MODEL_NAME,
                contents=prompt,
                config=generation_config,
            )
            try:
                response = future.result(timeout=TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                logging.error(f"Metadata Gemini API call timed out after {TIMEOUT_SECONDS} seconds")
                return {
                    "data": metadata_dict,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            except Exception as e:
                logging.error(f"Error during metadata Gemini API call execution: {e}")
                traceback.print_exc()
                return {
                    "data": metadata_dict,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }

        logging.info("Received metadata response from Gemini API.")

        # --- Retrieve Token Count ---
        try:
            if response.usage_metadata:
                prompt_tokens = response.usage_metadata.prompt_token_count
                completion_tokens = response.usage_metadata.candidates_token_count
                total_tokens = response.usage_metadata.total_token_count
                logging.info(f"Metadata Gemini Usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens} tokens")
            else:
                logging.warning("Usage metadata not found in metadata Gemini response.")
        except Exception as meta_error:
            logging.warning(f"Could not retrieve metadata token usage: {meta_error}")

        # --- Process Response ---
        raw_text = _parse_gemini_response_text(response)
        if not raw_text:
            logging.error("Failed to extract text from metadata Gemini response.")
            return {
                "data": metadata_dict,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }

        json_string = _extract_json_from_text(raw_text)
        if not json_string:
            logging.error("Could not extract JSON structure from metadata Gemini response.")
            return {
                "data": metadata_dict,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }

        try:
            metadata_dict = json.loads(json_string)
            logging.info("Successfully extracted metadata JSON.")
            # Ensure only expected keys are present (optional cleanup)
            allowed_keys = {"project_name", "project_number", "spec_section_number", "spec_section_title", "publication_date"}
            metadata_dict = {k: v for k, v in metadata_dict.items() if k in allowed_keys}

        except json.JSONDecodeError as e:
            logging.error(f"Extracted metadata text is not valid JSON: {e}")
            metadata_dict = {} # Reset to empty if parsing fails

    except ValueError as e:
        logging.error(f"Configuration error for metadata extraction: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred in call_gemini_for_metadata: {e}")
        traceback.print_exc()

    # Return dict containing extracted data and all token counts
    return {
        "data": metadata_dict,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    }


def generate_metadata(pdf_path: str) -> Dict[str, Any]:
    """
    Extracts metadata from the first page of a PDF using Gemini.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A dictionary containing:
        - 'data': The extracted metadata dictionary (or {} if failed).
        - 'prompt_tokens': Number of prompt tokens used.
        - 'completion_tokens': Number of completion tokens used.
        - 'total_tokens': Total tokens used.
        - 'model': The model name used.
    """
    logger = logging.getLogger(__name__)
    result = {
        "data": {},
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "model": MODEL_NAME
    }

    try:
        first_page_text = extract_first_page_text(pdf_path)
        if not first_page_text:
            logger.error(f"Could not extract text from the first page of {pdf_path}.")
            return result # Return default result with 0 tokens

        metadata_api_result = call_gemini_for_metadata(first_page_text)
        result["data"] = metadata_api_result.get("data", {})
        result["prompt_tokens"] = metadata_api_result.get("prompt_tokens", 0)
        result["completion_tokens"] = metadata_api_result.get("completion_tokens", 0)
        result["total_tokens"] = metadata_api_result.get("total_tokens", 0)

    except Exception as e:
        logger.error(f"Failed to generate metadata for {pdf_path}: {e}")
        traceback.print_exc()
        # Keep tokens as 0 if an error occurred during the process
        result["data"] = {}
        result["prompt_tokens"] = 0
        result["completion_tokens"] = 0
        result["total_tokens"] = 0

    return result


def extract_metadata_dict(pdf_path: str) -> dict:
    """
    Extracts metadata from the first page of a PDF using Gemini and returns as a dict.
    """
    result = generate_metadata(pdf_path)
    return result.get("data", {})


def main(pdf_path: str, output_filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Standalone execution: Extracts metadata, saves to JSON, and returns token info.
    (Kept for backward compatibility or direct script usage)
    """
    logger = logging.getLogger(__name__)
    if not output_filename:
        output_filename = os.path.splitext(pdf_path)[0] + "_metadata.json"

    # Call the core generation function
    metadata_result = generate_metadata(pdf_path)

    # Prepare the dictionary to be saved (including token counts)
    metadata_to_save = metadata_result.get("data", {}).copy() # Start with extracted data
    metadata_to_save["metadata_prompt_tokens"] = metadata_result.get("prompt_tokens", 0)
    metadata_to_save["metadata_completion_tokens"] = metadata_result.get("completion_tokens", 0)
    metadata_to_save["metadata_total_tokens"] = metadata_result.get("total_tokens", 0)

    # Save the metadata (including token counts) to JSON
    try:
        if output_filename:
             os.makedirs(os.path.dirname(output_filename), exist_ok=True)
             with open(output_filename, 'w') as f:
                 json.dump(metadata_to_save, f, indent=4)
             logger.info(f"Metadata successfully written to {output_filename}")
        else:
             logger.warning("No output filename provided, metadata JSON not saved.")
    except Exception as e:
        logger.error(f"Error saving metadata to {output_filename}: {e}")
        traceback.print_exc()

    # Return only the token info and model, as the original main did
    return {
        "prompt_tokens": metadata_result.get("prompt_tokens", 0),
        "completion_tokens": metadata_result.get("completion_tokens", 0),
        "total_tokens": metadata_result.get("total_tokens", 0),
        "model": metadata_result.get("model", MODEL_NAME),
        "data": metadata_result.get("data", {})
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(__file__)} <pdf_path> [output_json_path]")
        sys.exit(1)
    pdf_file_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    token_info = main(pdf_file_path, output_path)
    print(f"Metadata extraction finished. Info: {token_info}")
