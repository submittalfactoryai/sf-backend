import re
import json
import sys
import os
import logging
import argparse
from typing import Any, Dict, List, Optional, Tuple

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def get_line_level(line: str) -> Tuple[int, str, str]:
    """
    Determines the hierarchical level of a line based on its prefix.
    Returns:
        Tuple[int, str, str]: (level, key, content)
        level: Numerical level (0-5), or -1 for continuation text.
        key: The string to use as the dictionary key for this item.
        content: Any remaining text on the line after the key (currently unused but parsed).
    """
    stripped_line = line.strip() # Use stripped line for analysis

    # Level 0: Must be exact match
    if stripped_line == "PART 2 - PRODUCTS":
        return 0, stripped_line, ""

    # Level 1: digit.digit space text (Require space)
    match = re.match(r"^(\d+\.\d+)\s+(.+)", stripped_line)
    if match:
        return 1, stripped_line, "" # Key is the whole line

    # Level 2: UPPERCASE_LETTER. optional_space text
    match = re.match(r"^([A-Z])\.\s*(.+)", stripped_line)
    if match:
        return 2, stripped_line, "" # Key is the whole line

    # Level 3: digit. space text (Require space)
    match = re.match(r"^(\d+)\.\s+(.+)", stripped_line)
    if match:
        return 3, stripped_line, "" # Key is the whole line

    # Level 4: lowercase_letter. space text (Require space)
    match = re.match(r"^([a-z])\.\s+(.+)", stripped_line)
    if match:
        return 4, stripped_line, "" # Key is the whole line

    # Level 5: digit) space text (Require space)
    match = re.match(r"^(\d+)\)\s+(.+)", stripped_line)
    if match:
        return 5, stripped_line, "" # Key is the whole line

    # --- Add checks for prefix-only lines (treat as content) ---
    if re.match(r"^(\d+\.\d+)$", stripped_line): return -1, stripped_line, ""
    if re.match(r"^([A-Z])\.$", stripped_line): return -1, stripped_line, ""
    if re.match(r"^(\d+)\.$", stripped_line): return -1, stripped_line, ""
    if re.match(r"^([a-z])\.$", stripped_line): return -1, stripped_line, ""
    if re.match(r"^(\d+)\)$", stripped_line): return -1, stripped_line, ""

    # If none of the above, it's continuation text
    return -1, stripped_line, ""

# --- Core Parsing Logic ---

def parse_text_to_json_structure(text_content: str) -> Optional[Dict[str, Any]]:
    """
    Parses the structured text string into a nested JSON dictionary,
    including start and end content snippets.

    Args:
        text_content: The cleaned text string to parse.

    Returns:
        A dictionary representing the hierarchical structure, or None on error.
    """
    if not text_content:
        logging.warning("Received empty text_content for JSON parsing.")
        return None

    result: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = []
    last_item_info = {'parent_dict': None, 'key': None}
    line_num = -1

    try:
        # Get start/end snippets from the input string
        start_content = text_content[:100]
        end_content = text_content[-100:]

        result = {
            "start_content": start_content,
        }
        stack = [(-1, result)]

        lines = text_content.splitlines() # Split the input string into lines

        for line_num, line in enumerate(lines):
            stripped_line = line.strip()
            if not stripped_line:
                continue

            level, key, _ = get_line_level(stripped_line)
            logging.debug(f"L{line_num+1}: Level={level}, Key={repr(key)}, Line={repr(stripped_line)}")

            # Normalize the key to replace newlines with spaces
            # This helps prevent issues when newlines appear in keys within the JSON structure
            key = key.replace('\n', ' ').replace('\r', ' ')

            if level == -1:
                if last_item_info['parent_dict'] is not None and last_item_info['key'] is not None:
                    parent_dict = last_item_info['parent_dict']
                    old_key = last_item_info['key']
                    if old_key in parent_dict:
                        current_value = parent_dict[old_key]
                        separator = " " if not old_key.endswith(" ") and not stripped_line.startswith(" ") else ""
                        # Normalize continuation text too
                        normalized_line = stripped_line.replace('\n', ' ').replace('\r', ' ')
                        new_key = old_key + separator + normalized_line
                        logging.debug(f"  -> Cont.: Appending '{normalized_line}' to previous key '{old_key}' -> '{new_key}'")
                        del parent_dict[old_key]
                        parent_dict[new_key] = current_value
                        last_item_info['key'] = new_key
                    else:
                         logging.warning(f"  -> Cont.: Old key '{old_key}' not found in parent dictionary. Cannot append '{stripped_line}'. Ignoring.")
                else:
                    logging.warning(f"  -> Cont.: Found content line '{stripped_line}' but no previous item context. Ignoring.")
                continue

            while stack[-1][0] >= level:
                stack.pop()

            parent_dict = stack[-1][1]
            if not isinstance(parent_dict, dict):
                logging.error(f"  -> Structure Error: Expected parent at level {stack[-1][0]} to be a dict, but found {type(parent_dict)}. Cannot add key '{key}'. Skipping line {line_num+1}.")
                continue

            new_item: Dict[str, Any] = {}
            parent_dict[key] = new_item
            logging.debug(f"  -> New Item: Added key '{key}' at level {level} to parent level {stack[-1][0]}")

            last_item_info['parent_dict'] = parent_dict
            last_item_info['key'] = key
            stack.append((level, new_item))

        result["end_content"] = end_content

    except Exception as e:
        logging.error(f"Error processing text content at line {line_num+1}: {e}", exc_info=True)
        return None

    return result

def parse_text_file_to_json(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Reads a text file and parses it into a nested JSON dictionary.
    (Used for standalone script execution)
    """
    if not os.path.exists(filepath):
        logging.error(f"Input file not found: {filepath}")
        return None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            full_content = f.read()
        return parse_text_to_json_structure(full_content)
    except Exception as e:
        logging.error(f"Error reading file {filepath}: {e}", exc_info=True)
        return None

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Convert cleaned text file to JSON based on hierarchical structure.")
    parser.add_argument("input_file", help="Path to the cleaned text file (e.g., *_cleaned.txt).")
    parser.add_argument("-o", "--output", help="Path to the output JSON file (defaults to input filename with _final.json extension).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")

    # Basic argument check
    if len(sys.argv) < 2:
        parser.print_help(sys.stderr)
        sys.exit(1)
    # Handle '-h' or '--help' explicitly if needed, though argparse does this
    if sys.argv[1] in ['-h', '--help']:
         parser.print_help(sys.stderr)
         sys.exit(0)

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_filepath = args.input_file
    output_filepath = args.output
    if not output_filepath:
        base_name = os.path.splitext(input_filepath)[0]
        # Construct output name based on input (e.g., SEC2_cleaned.txt -> SEC2_final.json)
        if base_name.endswith('_cleaned'):
             base_name = base_name[:-8] # Remove '_cleaned'
        output_filepath = f"{base_name}_final.json"

    logging.info(f"Input file: {input_filepath}")
    logging.info(f"Output file: {output_filepath}")

    json_data = parse_text_file_to_json(input_filepath)

    if json_data:
        try:
            with open(output_filepath, 'w', encoding='utf-8') as f_out:
                # Use indent=4 for pretty printing
                json.dump(json_data, f_out, indent=4)
            logging.info(f"Successfully converted text to JSON: {output_filepath}")
        except IOError as e:
            logging.error(f"Error writing JSON to {output_filepath}: {e}")
        except Exception as e:
            logging.error(f"Error during JSON serialization: {e}", exc_info=True)
    else:
        logging.error("JSON conversion failed.")
        sys.exit(1) # Indicate failure

if __name__ == "__main__":
    main()