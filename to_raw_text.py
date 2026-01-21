"""
Extract raw text content from PDFs.
This module provides functionality to extract the raw text from a PDF file
without any additional processing or formatting.
"""

import os
import logging
from io import BytesIO
import fitz  # PyMuPDF

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_file_path=None, pdf_bytes=None):
    """
    Extract raw text from a PDF file.
    
    Args:
        pdf_file_path (str, optional): Path to the PDF file.
        pdf_bytes (bytes, optional): PDF file as bytes.
        
    Returns:
        str: Extracted text from the PDF.
        
    Raises:
        ValueError: If neither pdf_file_path nor pdf_bytes is provided.
    """
    if not pdf_file_path and not pdf_bytes:
        raise ValueError("Either pdf_file_path or pdf_bytes must be provided")
    
    try:
        # Open the PDF file
        if pdf_file_path:
            logger.info(f"Opening PDF from file: {pdf_file_path}")
            doc = fitz.open(pdf_file_path)
        else:
            logger.info("Opening PDF from bytes")
            doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
        
        # Extract text from each page
        text = ""
        page_count = doc.page_count
        logger.info(f"PDF has {page_count} pages")
        
        for page_num in range(page_count):
            logger.debug(f"Processing page {page_num + 1}/{page_count}")
            page = doc.load_page(page_num)
            page_text = page.get_text()
            text += page_text + "\n\n"
        
        logger.info(f"Successfully extracted text. Total characters: {len(text)}")
        return text
    
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise
    
    finally:
        # Close the document if it was opened
        if 'doc' in locals():
            doc.close()

def save_text_to_file(text, output_path):
    """
    Save extracted text to a file.
    
    Args:
        text (str): Text to save.
        output_path (str): Path to save the text file.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"Text saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving text to file: {str(e)}")
        return False

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract raw text from a PDF file')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--output', '-o', help='Output text file path')
    
    args = parser.parse_args()
    
    if os.path.exists(args.pdf_path):
        extracted_text = extract_text_from_pdf(pdf_file_path=args.pdf_path)
        
        # Determine output path
        output_path = args.output
        if not output_path:
            base_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
            output_path = f"{base_name}_raw_text.txt"
        
        # Save the text
        save_text_to_file(extracted_text, output_path)
        print(f"Text extracted and saved to {output_path}")
    else:
        print(f"Error: PDF file '{args.pdf_path}' not found")