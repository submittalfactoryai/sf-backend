#!/usr/bin/env python3
import os
import csv
import glob
import logging
from datetime import datetime
from .extract_with_llm import main as extract_main

# Cost rates (per 1M tokens, USD) for Gemini 1.5 Flash (<=128k tokens)
PROMPT_COST_PER_1M = 0.075
COMPLETION_COST_PER_1M = 0.30

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PDF_DIR = os.path.join(BASE_DIR, 'pdf-files')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'usage_log.csv')


def run_all():
    """
    Process all PDFs in PDF_DIR, extract product data, and log usage statistics.
    Skips files already processed (by filename) in usage_log.csv.
    """
    pdf_paths = glob.glob(os.path.join(PDF_DIR, '*.pdf'))
    if not pdf_paths:
        logging.warning(f"No PDF files found in {PDF_DIR}.")
        return

    # Ensure usage_log.csv exists with header if not present
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Timestamp',
                'Filename',
                'Filesize (Bytes)',
                'Input Tokens',
                'Output Tokens',
                'Model Used',
                'Estimated Cost (USD)'
            ])

    # Collect already processed filenames
    processed_filenames = set()
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                processed_filenames.add(row['Filename'])

    with open(LOG_FILE, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for pdf_path in pdf_paths:
            filename = os.path.basename(pdf_path)
            if filename in processed_filenames:
                logging.info(f"Skipping already processed file: '{filename}'")
                continue
            filesize = os.path.getsize(pdf_path)
            logging.info(f"Processing '{filename}' ({filesize} bytes)...")

            # Extract using LLM
            result = extract_main(pdf_path)
            prompt_tokens = result.get('prompt_tokens', 0)
            completion_tokens = result.get('completion_tokens', 0)
            model_used = result.get('model', '')

            # Estimate cost using Gemini 1.5 Flash pricing
            cost = (prompt_tokens / 1_000_000) * PROMPT_COST_PER_1M + (completion_tokens / 1_000_000) * COMPLETION_COST_PER_1M

            # Timestamp
            timestamp = datetime.now().isoformat()

            # Log row
            writer.writerow([
                timestamp,
                filename,
                filesize,
                prompt_tokens,
                completion_tokens,
                model_used,
                f"{cost:.6f}"
            ])
            logging.info(f"Logged stats for '{filename}': tokens={prompt_tokens + completion_tokens}, cost=${cost:.6f}")


def main():
    run_all()


if __name__ == '__main__':
    main()
