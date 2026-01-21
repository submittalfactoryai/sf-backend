import logging
import argparse
import os
import sys
import re
from typing import List, Optional

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Functions ---

def merge_split_lines(cleaned_lines: List[str]) -> List[str]:
    """
    Post-processes cleaned lines to merge lines where a heading/item number/letter
    is on one line and its text is on the next.
    """
    final_lines = []
    i = 0
    merge_end_pattern = r"^\s*(\d+(\.\d+)*|[a-zA-Z])\.?\s*$"
    merge_start_pattern = r"^\s*(\d+(\.\d+)*|[a-zA-Z])\."

    logging.debug("Starting merge_split_lines function.")
    while i < len(cleaned_lines):
        current_line = cleaned_lines[i]
        current_line_stripped = current_line.strip()
        logging.debug(f"Merge Check: Line {i}: {repr(current_line_stripped)}")

        match_end = re.match(merge_end_pattern, current_line_stripped)
        if match_end and (i + 1) < len(cleaned_lines):
            logging.debug(f"  -> Matched end pattern: {repr(match_end.group(0))}")
            next_line = cleaned_lines[i+1]
            next_line_stripped = next_line.strip()
            logging.debug(f"  -> Next line {i+1}: {repr(next_line_stripped)}")

            match_start_next = re.match(merge_start_pattern, next_line_stripped)
            if next_line_stripped and not match_start_next:
                logging.debug(f"  -> Next line is suitable for merging. Performing merge.")
                merged_line = current_line.rstrip() + " " + next_line.lstrip()
                if not merged_line.endswith('\n'):
                    merged_line += '\n'
                final_lines.append(merged_line)
                logging.debug(f"  -> Merged result: {repr(merged_line.strip())}")
                i += 2
            else:
                logging.debug(f"  -> Next line NOT suitable. Blank: {not next_line_stripped}. Starts like item: {bool(match_start_next)}")
                final_lines.append(current_line)
                i += 1
        else:
            if not match_end:
                logging.debug(f"  -> Did not match end pattern.")
            elif not (i + 1) < len(cleaned_lines):
                logging.debug(f"  -> No next line exists.")
            final_lines.append(current_line)
            i += 1

    logging.debug("Finished merge_split_lines function.")
    return final_lines


def clean_text(raw_text: str) -> Optional[str]:
    """
    Cleans the raw extracted text by collapsing blank lines and merging split lines.

    Args:
        raw_text: The raw text string to clean.

    Returns:
        The cleaned text string, or None if input is empty or an error occurs.
    """
    if not raw_text:
        logging.warning("Received empty raw_text for cleaning.")
        return None

    logging.info(f"Starting text cleaning process for input text (length: {len(raw_text)}).")

    lines = raw_text.splitlines(keepends=True)
    intermediate_cleaned_lines = []
    previous_line_was_blank = False

    try:
        for line_num, line in enumerate(lines, 1):
            stripped_line = line.strip()
            logging.debug(f"Processing Line {line_num}: {repr(stripped_line)}")

            is_blank = not stripped_line
            if is_blank:
                if not previous_line_was_blank:
                    logging.debug("  -> Keeping first blank line.")
                    intermediate_cleaned_lines.append(line)
                    previous_line_was_blank = True
                else:
                    logging.debug("  -> Skipping consecutive blank line.")
                    pass
            else:
                logging.debug("  -> Keeping content line.")
                intermediate_cleaned_lines.append(line)
                previous_line_was_blank = False

    except Exception as e:
        logging.error(f"Error during blank line collapsing: {e}", exc_info=True)
        return None

    logging.info(f"Performing post-processing merge of split lines.")
    final_cleaned_lines = merge_split_lines(intermediate_cleaned_lines)

    cleaned_text = "".join(final_cleaned_lines)
    logging.info(f"Successfully cleaned text. Output length: {len(cleaned_text)}")
    return cleaned_text


def clean_extracted_text_file(text_filepath: str) -> None:
    """
    Reads text from a file, cleans it using clean_text(),
    and writes the result to a new file.

    Args:
        text_filepath: Path to the raw extracted text file (e.g., *_to_text.txt).
    """
    logging.info(f"Starting text cleaning process for file: {text_filepath}")

    if not os.path.exists(text_filepath):
        logging.error(f"Input text file not found: {text_filepath}")
        return

    base_name = os.path.splitext(text_filepath)[0]
    if base_name.endswith('_to_text'):
        base_name = base_name[:-8]
    output_filepath = f"{base_name}_cleaned.txt"
    logging.info(f"Output will be written to: {output_filepath}")

    raw_text_content: Optional[str] = None
    try:
        with open(text_filepath, 'r', encoding='utf-8') as f_text:
            raw_text_content = f_text.read()
    except Exception as e:
        logging.error(f"Error reading text file {text_filepath}: {e}", exc_info=True)
        return

    if raw_text_content is None:
         logging.error(f"Failed to read content from {text_filepath}")
         return

    cleaned_content = clean_text(raw_text_content)

    if cleaned_content is not None:
        try:
            with open(output_filepath, 'w', encoding='utf-8') as f_out:
                f_out.write(cleaned_content)
            logging.info(f"Successfully cleaned text written to {output_filepath}")
        except IOError as e:
            logging.error(f"Error writing cleaned text to {output_filepath}: {e}")
    else:
        logging.warning(f"Cleaning process returned no content for {text_filepath}. No output file generated.")


# --- Main Execution Block ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean extracted text by collapsing blank lines and merging split lines."
    )
    parser.add_argument(
        "text_file",
        help="Path to the raw text file to be cleaned (e.g., input_to_text.txt)."
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    clean_extracted_text_file(args.text_file)