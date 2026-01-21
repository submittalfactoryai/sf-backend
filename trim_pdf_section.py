from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response
import tempfile
import shutil
import re
import os
import logging
import io
from pypdf import PdfReader, PdfWriter

router = APIRouter()

# Flexible regex patterns (case-insensitive, allow spaces, hyphens, colons)
PART_2_REGEX = r"PART\s*[-:]?\s*2\s*[-:]?\s*PRODUCTS"
PART_3_REGEX = r"PART\s*[-:]?\s*3\s*[-:]?\s*EXECUTION"

@router.post("/trim-pdf-part2")
async def trim_pdf_part2(file: UploadFile = File(...)):
    """
    Receives a PDF, finds PART 2 and PART 3 section headers, and returns a new PDF containing only PART 2 section pages.
    """
    temp_pdf_path = None
    trimmed_pdf_bytes = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file:
            temp_pdf_path = temp_pdf_file.name
            shutil.copyfileobj(file.file, temp_pdf_file)

        reader = PdfReader(temp_pdf_path)
        part2_pattern = re.compile(PART_2_REGEX, re.IGNORECASE)
        part3_pattern = re.compile(PART_3_REGEX, re.IGNORECASE)
        part2_page, part3_page = None, None

        # Find the start and end pages for PART 2
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if part2_page is None and part2_pattern.search(text):
                part2_page = i
            if part3_page is None and part3_pattern.search(text):
                part3_page = i
            # Optimization: Stop searching for part3 if part2 is already found and part3 is found on a later page
            if part2_page is not None and part3_page is not None and part3_page >= part2_page:
                break
            # Optimization: If part2 is found, no need to keep searching for it
            # Continue searching for part3 even if part2 is found

        if part2_page is None:
            raise HTTPException(status_code=400, detail="PART 2 section not found in PDF.")

        # Determine the index of the last page to include in the trimmed section
        # If PART 3 is found, include the page it starts on.
        # If PART 3 is not found, include up to the last page of the document.
        last_page_index_to_include = part3_page if part3_page is not None else (len(reader.pages) - 1)

        # Validate that the identified end page is not before the start page
        if last_page_index_to_include < part2_page:
            # This case implies PART 3 was found, but on a page before PART 2
            raise HTTPException(status_code=400, detail="PART 3 found on a page before PART 2. Cannot trim correctly.")

        # Create trimmed PDF in memory
        writer = PdfWriter()

        # Loop from the start page up to and INCLUDING the last page index
        # The range's stop parameter is exclusive, so add 1
        for i in range(part2_page, last_page_index_to_include + 1):
            # Ensure index is valid (should be guaranteed by logic above, but as safeguard)
            if 0 <= i < len(reader.pages):
                writer.add_page(reader.pages[i])
            else:
                # This shouldn't happen with the current logic, but log if it does
                logging.warning(f"Attempted to add invalid page index {i} while trimming. Total pages: {len(reader.pages)}")

        # Write to a BytesIO buffer
        pdf_buffer = io.BytesIO()
        writer.write(pdf_buffer)
        pdf_buffer.seek(0)
        trimmed_pdf_bytes = pdf_buffer.getvalue()

        if not trimmed_pdf_bytes:
            raise HTTPException(status_code=500, detail="Failed to create trimmed PDF content (empty result).")

        # Return the bytes directly using Response
        return Response(
            content=trimmed_pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'inline; filename="trimmed_{file.filename}"'
            }
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.error(f"Error trimming PDF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)