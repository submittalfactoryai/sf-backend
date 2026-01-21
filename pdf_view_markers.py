from fastapi import APIRouter, UploadFile, File, HTTPException
from pypdf import PdfReader
import tempfile
import shutil
import re

router = APIRouter()

# Flexible regex patterns for section headers (copied from to_text.py)
PART_2_START_REGEX = r"PART\s*[-:]?\s*2\s*[-:]?\s*PRODUCTS"
PART_3_START_REGEX = r"PART\s*[-:]?\s*3\s*[-:]?\s*EXECUTION"

@router.post("/api/pdf-section-pages")
async def get_section_page_numbers(file: UploadFile = File(...)):
    """
    Returns the page numbers (0-based) where 'PART 2 - Products' and 'PART 3 - Execution' appear in the uploaded PDF.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_pdf_path = f"{temp_dir}/{file.filename}"
        try:
            with open(temp_pdf_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            reader = PdfReader(temp_pdf_path)
            part2_page, part3_page = None, None
            part2_pattern = re.compile(PART_2_START_REGEX, re.IGNORECASE)
            part3_pattern = re.compile(PART_3_START_REGEX, re.IGNORECASE)
            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                if part2_page is None and part2_pattern.search(text):
                    part2_page = i
                if part3_page is None and part3_pattern.search(text):
                    part3_page = i
            return {"part2_page": part2_page, "part3_page": part3_page}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
