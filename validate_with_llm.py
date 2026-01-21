"""
Validate PDF content against product specifications using LLM.
This module provides functionality to validate if a PDF contains the technical
specifications for a specific product using a language model.
"""

import os
import json
import logging
import time
import re
from typing import Dict, List, Any, Optional, Tuple
import google.generativeai as genai
from dotenv import load_dotenv

# Import the PDF text extraction utility
try:
    from to_raw_text import extract_text_from_pdf
except ImportError:
    from .to_raw_text import extract_text_from_pdf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Google Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY not found in environment variables")
genai.configure(api_key=GEMINI_API_KEY)

# Select the model to use for validation
MODEL_NAME = "gemini-2.0-flash"

EXAMPLE_JSON = '''{
  "valid": "Yes",
  "validation_score": 60,
  "product_name_found": "Yes",
  "specifications_match": "3/5",
  "matched_specifications": ["Original Specification Text for Spec 1 if found", "Original Specification Text for Spec 2 if found", "Original Specification Text for Spec 3 if found"],
  "unmatched_specifications": ["Original Specification Text for Spec 4 if not found", "Original Specification Text for Spec 5 if not found"],
  "any_manufacturer_found": "Yes",
  "found_manufacturers": ["Manufacturer A (if listed in product and found in PDF)"],
  "unmatched_manufacturers": ["Manufacturer B (if listed in product but not found in PDF)"],
  "summary": "The product name and several specifications were confirmed. Manufacturer A was identified from the provided list. However, [Spec 4] and [Spec 5] could not be verified, and Manufacturer B was not found. The document appears to be a relevant data sheet for the specified product."
}'''

def format_product_for_prompt(product: Dict[str, Any]) -> str:
    """
    Format product data for inclusion in the LLM prompt.
    
    Args:
        product (Dict[str, Any]): Product data with name and specifications
        
    Returns:
        str: Formatted product information for the prompt
    """
    formatted = f"Product Name: {product.get('name', 'Unknown')}\n\n"
    
    if specs := product.get('specifications', []):
        formatted += "Technical Specifications:\n"
        for i, spec in enumerate(specs, 1):
            formatted += f"{i}. {spec}\n"
    else:
        formatted += "No technical specifications provided.\n"
    
    if manufacturers := product.get('manufacturers', []):
        formatted += "\nManufacturers:\n"
        for i, manufacturer in enumerate(manufacturers, 1):
            formatted += f"{i}. {manufacturer}\n"
    
    return formatted

def create_validation_prompt(
    product: Dict[str, Any], 
    pdf_text: str, 
    text_sample_length: int = 5000
) -> str:
    """
    Create a prompt for the LLM to validate if the PDF contains the specified product.
    
    Args:
        product (Dict[str, Any]): Product data with name and specifications
        pdf_text (str): The extracted text from the PDF
        text_sample_length (int): Maximum length of PDF text to include in prompt
        
    Returns:
        str: Formatted prompt for the LLM
    """
    # Truncate the PDF text to not exceed token limits
    text_sample = pdf_text[:text_sample_length]
    if len(pdf_text) > text_sample_length:
        text_sample += f"\n\n[Text truncated, {len(pdf_text) - text_sample_length} more characters available...]"
    
    formatted_product = format_product_for_prompt(product)
    
    prompt = f"""
You are a technical specification validation assistant. Your task is to validate if a construction specification PDF document contains the technical specifications for a given product.

### Product Details:
{formatted_product}

### Sample from PDF Document:
```
{text_sample}
```

### Validation Instructions:
Your goal is to determine if the 'Sample from PDF Document' is a suitable data sheet for the product described in 'Product Details'.

1.  **Product Name Check**:
    *   Is the `Product Name` from 'Product Details' (or a very close, unambiguous equivalent) explicitly mentioned in the 'Sample from PDF Document'?
    *   Set `product_name_found` to "Yes" or "No".

2.  **Technical Specifications Check**:
    *   Carefully review each technical specification listed under 'Technical Specifications' in 'Product Details'.
    *   For each one, try to find a matching statement in the 'Sample from PDF Document'. A match means the PDF confirms the specification, even if worded slightly differently, as long as the technical meaning and values are equivalent.
    *   Count how many specifications from 'Product Details' are confirmed (`X`) out of the total number of specifications listed (`Y`).
    *   Set `specifications_match` to "X/Y".
    *   **Crucially**: For every specification from 'Product Details' that you *can* confirm in the PDF, add its **original text** (exactly as it appears in 'Product Details') to the `matched_specifications` array. If no specifications are confirmed, this array MUST be empty (`[]`).
    *   **Crucially**: For every specification from 'Product Details' that you *cannot* confirm in the PDF, add its **original text** (exactly as it appears in 'Product Details') to the `unmatched_specifications` array. If all specifications are confirmed, this array MUST be empty (`[]`).

3.  **Manufacturers Check**:
    *   Review the list of 'Manufacturers' in 'Product Details'.
    *   Determine if *at least one* of these manufacturers is mentioned in the 'Sample from PDF Document'.
    *   Set `any_manufacturer_found` to "Yes" or "No".
    *   Populate the `found_manufacturers` array with the names of all manufacturers from 'Product Details' that *are* mentioned in the PDF. If none are found, this array MUST be empty (`[]`).
    *   Populate the `unmatched_manufacturers` array with the names of all manufacturers from 'Product Details' that *are not* mentioned in the PDF. If all are found, this array MUST be empty (`[]`).

4.  **Overall Validity**:
    *   Based on your findings (product name, specs, manufacturers), decide if the PDF is a valid data sheet for the product. Consider if enough critical information is present.
    *   Set `valid` to "Yes" or "No".

5.  **Validation Score**:
    *   Calculate a `validation_score` (an integer between 0 and 100). This score should reflect your overall confidence.
        *   100 means a perfect match: product name found, all specifications confirmed, at least one manufacturer confirmed (if manufacturers were listed in product details) and all listed manufacturers found.
        *   Lower scores indicate poorer matches (e.g., some specs missing, no listed manufacturers found, product name ambiguous).
        *   0 means little to no relevance.

6.  **Summary**:
    *   Provide a concise `summary` (2-3 sentences) of your findings. Explain the basis for your `valid` status and `validation_score`. Specifically mention key confirmed or unconfirmed details. Ensure the summary is written in a professional and objective tone.

Remember to fill all fields in the JSON structure provided in the example, using your actual findings.

### Response Format:
Respond ONLY in valid JSON. Fill in the fields below with your actual findings from the PDF and product data. DO NOT copy the example or use placeholders. If you do, your answer will be rejected.

The JSON must look like this (but with real values):

{EXAMPLE_JSON}

Do not include any explanation or text outside the JSON.
"""
    return prompt

def extract_json_from_llm(text: str) -> Optional[dict]:
    """
    Extracts and parses JSON from LLM output, handling code blocks and common issues.
    """
    # Remove markdown code block if present
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.DOTALL)
    json_str = match.group(1) if match else text
    json_str = json_str.strip()
    
    # Try parsing the original JSON first
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Initial JSON parse failed: {e}. Attempting to fix common issues...")
        
        # Attempt to fix common issues
        fixed = json_str
        
        # Fix mixed quotes - ensure consistent double quotes for JSON keys and values
        # Replace patterns like '"key': with '"key":
        fixed = re.sub(r'"([^"]+)\':\s*', r'"\1": ', fixed)
        # Replace patterns like "'key": with '"key":
        fixed = re.sub(r'\'([^\']+)":\s*', r'"\1": ', fixed)
        
        # Replace single quotes around values with double quotes
        # This is more complex as we need to avoid replacing apostrophes within text
        # For now, handle simple cases where single quotes wrap entire values
        fixed = re.sub(r':\s*\'([^\']*?)\'\s*([,\}\]])', r': "\1"\2', fixed)
        
        # Replace any remaining single quotes with double quotes (as last resort)
        # But be careful not to replace apostrophes within words
        # Only replace single quotes that are clearly meant as JSON delimiters
        fixed = re.sub(r"([^a-zA-Z])'([^']*?)'([^a-zA-Z])", r'\1"\2"\3', fixed)
        
        # Remove trailing commas before closing braces/brackets
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
        
        # Try parsing the fixed version
        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e2:
            logger.warning(f"Fixed JSON parse also failed: {e2}. Attempting manual cleanup...")
            
            # More aggressive fixes
            # Replace common problematic patterns
            fixed = fixed.replace("'", '"')  # Replace all single quotes as last resort
            
            # Try to fix escaped quotes issues
            fixed = fixed.replace('\\"', '"')
            
            # Remove any remaining trailing commas
            fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
            
            try:
                return json.loads(fixed)
            except json.JSONDecodeError as e3:
                logger.error(f"All JSON parsing attempts failed. Original error: {e}, Fixed error: {e2}, Final error: {e3}")
                logger.error(f"Failed to parse LLM output as JSON. Raw output: {text}")
                logger.error(f"Final attempted fix: {fixed}")
                return None

def validate_pdf_with_llm(
    pdf_path: Optional[str] = None,
    pdf_bytes: Optional[bytes] = None,
    product_data: Dict[str, Any] = None,
    temperature: float = 0.1,
    max_output_tokens: int = 2048
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate if a PDF contains specifications for a given product using an LLM.
    Now returns token usage and model info in the result dictionary.

    Args:
        pdf_path (str, optional): Path to the PDF file.
        pdf_bytes (bytes, optional): PDF file as bytes.
        product_data (Dict[str, Any]): Product data to validate against.
        temperature (float): Temperature setting for LLM.
        max_output_tokens (int): Maximum output tokens for LLM response.
        
    Returns:
        Tuple[bool, Dict[str, Any]]: (success, result_data)
    """
    # Validate inputs
    if not pdf_path and not pdf_bytes:
        logger.error("Either pdf_path or pdf_bytes must be provided")
        return False, {"error": "No PDF provided"}
    
    if not product_data:
        logger.error("Product data must be provided")
        return False, {"error": "No product data provided"}
    
    try:
        # Extract text from PDF
        logger.info("Extracting text from PDF")
        pdf_text = extract_text_from_pdf(pdf_file_path=pdf_path, pdf_bytes=pdf_bytes)
        
        if not pdf_text.strip():
            logger.error("Extracted PDF text is empty")
            return False, {"error": "Extracted PDF text is empty"}
        
        # Create the validation prompt
        prompt = create_validation_prompt(product_data, pdf_text)
        
        # Initialize the model
        logger.info(f"Using {MODEL_NAME} for validation")
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            }
        )
        
        # Get response from LLM
        logger.info("Sending validation request to LLM")
        start_time = time.time()
        response = model.generate_content(prompt)
        elapsed_time = time.time() - start_time
        response_text = response.text
        logger.info(f"Raw LLM response before parsing: {response_text}")
        
        # Attempt to get token counts
        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            if hasattr(response.usage_metadata, 'prompt_token_count'):
                prompt_tokens = response.usage_metadata.prompt_token_count
            if hasattr(response.usage_metadata, 'candidates_token_count'): # For genai.GenerativeModel
                completion_tokens = response.usage_metadata.candidates_token_count
            elif hasattr(response.usage_metadata, 'completion_token_count'): # Alternative name
                completion_tokens = response.usage_metadata.completion_token_count
            logger.info(f"Token usage: Prompt={prompt_tokens}, Completion={completion_tokens}")
        else:
            logger.warning("Usage metadata not found in LLM response for validation.")

        parsed = extract_json_from_llm(response_text)
        
        # Base result structure, including fields always present
        result_dict = {
            "response": response_text, # Raw LLM response text
            "elapsed_time": elapsed_time,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "model_used": MODEL_NAME, # Using the globally defined MODEL_NAME for this module
            # Default values matching EXAMPLE_JSON structure for robustness
            "valid": "No",
            "validation_score": 0,
            "product_name_found": "No",
            "specifications_match": "0/0",
            "matched_specifications": [],
            "unmatched_specifications": [],
            "any_manufacturer_found": "No",
            "found_manufacturers": [],
            "unmatched_manufacturers": [],
            "summary": "Processing failed or LLM response was not parsable."
        }
        
        if parsed:
            # Update result_dict with successfully parsed fields
            result_dict.update({
                "valid": parsed.get("valid", "No"),
                "validation_score": int(parsed.get("validation_score", 0)),
                "product_name_found": parsed.get("product_name_found", "No"),
                "specifications_match": parsed.get("specifications_match", "0/0"),
                "matched_specifications": parsed.get("matched_specifications", []),
                "unmatched_specifications": parsed.get("unmatched_specifications", []),
                "any_manufacturer_found": parsed.get("any_manufacturer_found", "No"),
                "found_manufacturers": parsed.get("found_manufacturers", []),
                "unmatched_manufacturers": parsed.get("unmatched_manufacturers", []),
                "summary": parsed.get("summary", "Summary not provided by LLM.")
            })
            
            # Refined Warning Logic
            specs_match_str = result_dict["specifications_match"]
            any_manufacturer_is_found = result_dict["any_manufacturer_found"].lower() == "yes"
            found_manufacturers_list = result_dict["found_manufacturers"]
            unmatched_manufacturers_list = result_dict["unmatched_manufacturers"]
            
            current_validation_score = result_dict["validation_score"]
            product_name_is_found = result_dict["product_name_found"].lower() == "yes"
            
            specs_matched, specs_total = 0,0
            
            try:
                if '/' in specs_match_str: specs_matched, specs_total = map(int, specs_match_str.split('/'))
            except (ValueError, AttributeError):
                logger.warning(f"Could not parse specs_match string: '{specs_match_str}'. Using defaults for warning logic.")

            # Manufacturer counts from product_data (if available, otherwise infer from lists)
            # This requires product_data to be available here, or make assumptions.
            # For now, inferring from LLM's output lists for simplicity in this block.
            num_total_manufacturers_from_product = len(found_manufacturers_list) + len(unmatched_manufacturers_list)
            
            warning_messages = []

            if specs_total > 0 and specs_matched < specs_total and not result_dict["unmatched_specifications"]:
                warning_messages.append(
                    f"specifications_match is '{specs_match_str}' (indicating some specs not matched) but 'unmatched_specifications' list is empty."
                )
            if specs_total > 0 and specs_matched > 0 and not result_dict["matched_specifications"]:
                 warning_messages.append(
                    f"specifications_match is '{specs_match_str}' (indicating some specs matched) but 'matched_specifications' list is empty."
                )

            if any_manufacturer_is_found and not found_manufacturers_list:
                warning_messages.append(
                    "'any_manufacturer_found' is Yes, but 'found_manufacturers' list is empty."
                )
            if not any_manufacturer_is_found and found_manufacturers_list:
                 warning_messages.append(
                    "'any_manufacturer_found' is No, but 'found_manufacturers' list is not empty."
                )
            
            # If manufacturers were expected (i.e., product_data had them), then unmatched_manufacturers should be populated if not all found.
            # This is a bit tricky without direct access to product_data.manufacturers here.
            # Simplified: if any_manufacturer_is_found is Yes, and there are manufacturers in product_data (implied if unmatched_manufacturers is not empty or found_manufacturers is not empty),
            # but unmatched_manufacturers is empty when it shouldn't be.
            # This logic is complex without product_data. Let's rely on the LLM for now or simplify the warning.
            # Example: If product_data has [M1, M2, M3] and LLM finds M1.
            # any_manufacturer_found = Yes, found_manufacturers = [M1], unmatched_manufacturers = [M2, M3]
            if num_total_manufacturers_from_product > 0: # Implies manufacturers were listed in product data
                if any_manufacturer_is_found and len(found_manufacturers_list) < num_total_manufacturers_from_product and not unmatched_manufacturers_list:
                     warning_messages.append(
                        f"'any_manufacturer_found' is Yes and not all manufacturers were found (found: {len(found_manufacturers_list)}/{num_total_manufacturers_from_product}), but 'unmatched_manufacturers' list is empty."
                    )
                if not any_manufacturer_is_found and num_total_manufacturers_from_product > 0 and not unmatched_manufacturers_list:
                    warning_messages.append(
                        f"'any_manufacturer_found' is No, but 'unmatched_manufacturers' list is empty (should list all {num_total_manufacturers_from_product} manufacturers from product data)."
                    )


            if current_validation_score == 100 and (
                not product_name_is_found or
                (specs_total > 0 and specs_matched < specs_total) or
                (num_total_manufacturers_from_product > 0 and (not any_manufacturer_is_found or unmatched_manufacturers_list)) # If manu listed, at least one must be found AND all must be found for score 100
            ):
                warning_messages.append(
                    f"validation_score is 100, but product_name_found is '{result_dict['product_name_found']}', specs_match is '{specs_match_str}', any_manufacturer_found is '{result_dict['any_manufacturer_found']}', or unmatched_manufacturers is not empty ('{unmatched_manufacturers_list}'), indicating an incomplete match for a perfect score."
                )
            
            if current_validation_score < 70 and product_name_is_found and \
               (specs_total == 0 or specs_matched == specs_total) and \
               (num_total_manufacturers_from_product == 0 or (any_manufacturer_is_found and not unmatched_manufacturers_list)): # if manu listed, one found and no unmatched
                warning_messages.append(
                    f"validation_score is low ({current_validation_score}), but product name, all specs (if any), and all manufacturers (if any were listed and found) appear matched. Mfrs found: {any_manufacturer_is_found}, unmatched mfrs: {unmatched_manufacturers_list}."
                )
            
            if warning_messages:
                logger.warning(
                    f"LLM response consistency checks failed: {'; '.join(warning_messages)}. Raw response: {response_text}"
                )
            
            return True, result_dict
        else: # Not parsed
            logger.error(f"Failed to parse LLM output as JSON. Raw response: {response_text}")
            # result_dict already contains default error values and raw response
            result_dict["summary"] = "Failed to parse LLM response as JSON." # Ensure summary reflects parsing error
            return False, result_dict
    except Exception as e:
        logger.error(f"Error validating PDF with LLM: {str(e)}")
        # Return a structured error matching the expected format as much as possible
        error_response = {
            "valid": "No", "validation_score": 0, "product_name_found": "No",
            "specifications_match": "0/0", 
            "matched_specifications": [], 
            "unmatched_specifications": [],
            "any_manufacturer_found": "No",
            "found_manufacturers": [],
            "unmatched_manufacturers": [],
            "summary": f"An exception occurred: {str(e)}",
            "error": str(e), # Include the specific error message
            "response": None, # No LLM response if exception before LLM call
            "elapsed_time": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "model_used": MODEL_NAME
        }
        return False, error_response

def validate_from_file(pdf_path: str, product_json_path: str) -> Dict[str, Any]:
    """
    Validate a PDF file against product specifications from a JSON file.
    
    Args:
        pdf_path (str): Path to the PDF file.
        product_json_path (str): Path to the JSON file containing product data.
        
    Returns:
        Dict[str, Any]: Validation result
    """
    # Load product data from JSON file
    try:
        with open(product_json_path, 'r') as f:
            product_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading product data from {product_json_path}: {str(e)}")
        return {"success": False, "error": f"Error loading product data: {str(e)}"}
    
    # Validate the PDF
    success, result = validate_pdf_with_llm(pdf_path=pdf_path, product_data=product_data)
    
    # Add pdf_path and product_json_path to the output for context
    # Ensure these are added outside the main result dict from LLM to avoid overriding
    final_result = result.copy() # Start with LLM results
    final_result['pdf_path_validated'] = pdf_path
    final_result['product_json_used'] = product_json_path
    
    if success:
        logger.info("Validation successful.")
    else:
        logger.error("Validation failed or an error occurred.")
        
    return final_result

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate PDF against product JSON using LLM")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("product_json_path", help="Path to the product JSON file")
    args = parser.parse_args()
    
    result = validate_from_file(args.pdf_path, args.product_json_path)
    print(json.dumps(result, indent=2))