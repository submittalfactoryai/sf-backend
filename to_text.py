import fitz  # PyMuPDF
import sys
import os
import re
import logging
from typing import Optional

# --- Configuration & Constants ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define more flexible regex patterns as constants
# Allows for variations in spacing (zero or more), hyphens, colons, and case.
# Matches "PART", optional space/hyphen/colon, "2", optional space-hyphen/colon, "PRODUCTS"
PART_2_START_REGEX = r"PART\s*[-:]?\s*2\s*[-:]?\s*PRODUCTS?"
# Matches "PART", optional space-hyphen/colon, "3", optional space-hyphen/colon, "EXECUTION"
PART_3_START_REGEX = r"PART\s*[-:]?\s*3\s*[-:]?\s*EXECUTION"

# --- New Function for PART 2 Validation ---

def check_part2_exists(pdf_path: str) -> bool:
    """
    Checks if "PART 2 - PRODUCTS" section exists in the PDF.
    
    Args:
        pdf_path: The path to the PDF file.
        
    Returns:
        True if PART 2 - PRODUCTS is found, False otherwise.
    """
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file not found at: {pdf_path}")
        return False

    try:
        with fitz.open(pdf_path) as doc:
            logging.info(f"Checking for PART 2 - PRODUCTS in {doc.page_count} pages from PDF: {os.path.basename(pdf_path)}")
            full_text = ""
            for idx, page in enumerate(doc):
                page_text = page.get_text("text")
                full_text += page_text + "\n"

        # Compile regex with IGNORECASE, MULTILINE, and DOTALL flags for robustness
        part2_regex = re.compile(PART_2_START_REGEX, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        
        part2_match = part2_regex.search(full_text)
        if part2_match:
            logging.info(f"PART 2 - PRODUCTS found in {os.path.basename(pdf_path)} at position {part2_match.start()}")
            return True
        else:
            logging.warning(f"PART 2 - PRODUCTS not found in {os.path.basename(pdf_path)}")
            return False

    except fitz.FitzError as e:
        logging.error(f"PyMuPDF (Fitz) error processing {pdf_path}: {e}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred while checking PART 2 in {pdf_path}: {e}", exc_info=True)
        return False

# --- Core Function ---

def extract_text_between_patterns(pdf_path: str, start_pattern_regex: str, end_pattern_regex: str) -> Optional[str]:
    """
    Extracts text from a PDF between two specified regex patterns.

    Args:
        pdf_path: The path to the PDF file.
        start_pattern_regex: The regex pattern marking the start of the desired text.
        end_pattern_regex: The regex pattern marking the end of the desired text.

    Returns:
        The extracted text between the patterns, or None if patterns aren't found
        correctly or an error occurs.
    """
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file not found at: {pdf_path}")
        return None

    try:
        with fitz.open(pdf_path) as doc:
            logging.info(f"Reading {doc.page_count} pages from PDF: {os.path.basename(pdf_path)}")
            full_text = ""
            for idx, page in enumerate(doc):
                page_text = page.get_text("text")
                logging.debug(f"Page {idx+1}: {repr(page_text[:100])}...")
                full_text += page_text + "\n"

        # Compile regex with IGNORECASE, MULTILINE, and DOTALL flags for robustness
        start_regex = re.compile(start_pattern_regex, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        end_regex = re.compile(end_pattern_regex, re.IGNORECASE | re.MULTILINE | re.DOTALL)

        start_match = start_regex.search(full_text)
        if not start_match:
            logging.warning(f"Start pattern ({start_pattern_regex}) not found in {os.path.basename(pdf_path)}.")
            return None
        logging.info(f"Found start pattern at index {start_match.start()} to {start_match.end()}: {repr(full_text[start_match.start():start_match.end()])}")
        start_index = start_match.start()  # Start extracting *from* the start pattern

        # Search for the end pattern *after* the start pattern's match
        end_match = end_regex.search(full_text, pos=start_index + 1) # Start search after the start pattern match begins
        if not end_match:
            logging.warning(f"End pattern ({end_pattern_regex}) not found after start pattern in {os.path.basename(pdf_path)}. Returning text to end of document.")
            end_index = len(full_text)
        else:
            logging.info(f"Found end pattern at index {end_match.start()} to {end_match.end()}: {repr(full_text[end_match.start():end_match.end()])}")
            end_index = end_match.start()

        extracted_text = full_text[start_index:end_index].strip()
        logging.info(f"Successfully extracted text between patterns from {os.path.basename(pdf_path)}. Extracted length: {len(extracted_text)} characters.")
        return extracted_text

    except fitz.FitzError as e:
        logging.error(f"PyMuPDF (Fitz) error processing {pdf_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while processing {pdf_path}: {e}", exc_info=True)
        return None

def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Wrapper for extract_text_between_patterns to provide compatibility with existing imports.
    Extracts text between PART 2 and PART 3 sections in the PDF.
    """
    return extract_text_between_patterns(pdf_path, PART_2_START_REGEX, PART_3_START_REGEX)

def split_text_by_part(text: str):
    """
    Splits the text into parts based on 'PART X -' style section headers.
    Returns a list of parts (including the header in each part).
    """
    import re
    # Regex to match 'PART' followed by a number and a dash or colon
    pattern = re.compile(r'(PART\s*[-:]?\s*\d+\s*[-:]?\s*[A-Z ]+)', re.IGNORECASE)
    matches = list(pattern.finditer(text))
    if not matches:
        return [text]
    parts = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        parts.append(text[start:end].strip())
    return parts

# --- Main Execution Block ---

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(__file__)} <pdf_file_path>")
        sys.exit(1)

    pdf_file_path = sys.argv[1]

    logging.info(f"Starting text extraction process for PDF: {pdf_file_path}")

    part2_content = extract_text_between_patterns(pdf_file_path, PART_2_START_REGEX, PART_3_START_REGEX)

    if part2_content is not None: # Check explicitly for None, as empty string could be valid
        output_filename = os.path.splitext(pdf_file_path)[0] + "_to_text.txt"
        try:
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(part2_content)
            logging.info(f"Successfully extracted PART 2 content to {output_filename}")
        except IOError as e:
            logging.error(f"Failed to write extracted text to {output_filename}: {e}")
            sys.exit(1) # Exit if we can't write the output
    else:
        logging.warning(f"Could not extract PART 2 content from {pdf_file_path}. No output file generated.")
        # Optionally exit with a different code if extraction failure is critical
        # sys.exit(2)
