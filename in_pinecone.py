#!/usr/bin/env python3
"""
in_pinecone.py

CLI script to verify if a PDF section (using Project_Number and Section_Number metadata)
has already been indexed in Pinecone. Metadata is generated on-the-fly via metadata.py.
"""
import argparse
import os
import sys

from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone.exceptions import PineconeApiException as ApiException
try:
    from metadata import extract_metadata_dict  # Import the new function
except ImportError:
    from .metadata import extract_metadata_dict  # Import the new function

# Load environment variables (.env) early
load_dotenv()

# Fallback vector dimension if index.describe_index_stats() fails
DEFAULT_DIMENSION = 768


def sanitize_index_name(project_number: str) -> str:
    """
    Convert a project number into a valid Pinecone index name.
    Lowercase, replace spaces/underscores/dots with hyphens.
    """
    # Replace dots with hyphens as well
    return str(project_number).strip().lower().replace('_', '-').replace(' ', '-').replace('.', '-')


def list_existing_indexes(client: Pinecone) -> list[str]:
    """
    Retrieve existing index names, handling different client versions.
    """
    response = client.list_indexes()
    if hasattr(response, 'names'):
        names_attr = response.names
        return names_attr() if callable(names_attr) else names_attr
    return response


def check_in_pinecone(metadata: dict) -> bool:
    """
    Check if the section identified by metadata exists in the Pinecone index named
    after the Project_Number, within the namespace derived from Section_Number.

    Returns:
        True if found, False otherwise.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("Error: PINECONE_API_KEY not set.")
        return False

    client = Pinecone(api_key=api_key)
    project_num = metadata.get("Project_Number")
    section_num = metadata.get("Section_Number")

    if not project_num or not section_num:
        print("Error: metadata missing 'Project_Number' or 'Section_Number'.")
        return False

    index_name = sanitize_index_name(project_num)
    # Derive namespace from Section_Number, matching to_pinecone.py and retrieve.py
    namespace = str(section_num).replace(' ', '-').replace('/', '_')
    print(f"Checking index '{index_name}' (namespace: '{namespace}') for section '{section_num}'...")

    # Verify index existence
    existing = list_existing_indexes(client)
    if index_name not in existing:
        print(f"Index '{index_name}' not found.")
        return False

    index = client.Index(index_name)
    # Determine vector dimension
    try:
        stats = index.describe_index_stats()
        dimension = stats.dimension or DEFAULT_DIMENSION
        print(f"Detected index dimension: {dimension}")
    except Exception:
        print(f"Warning: using fallback dimension {DEFAULT_DIMENSION}")
        dimension = DEFAULT_DIMENSION

    # Query with filter *and* namespace
    response = index.query(
        namespace=namespace,  # Add the namespace here
        vector=[0.0] * dimension,
        filter={"Section_Number": {"$eq": section_num}},  # Keep filter for potential robustness
        top_k=1,
        include_metadata=False
    )

    matches = bool(getattr(response, 'matches', response.get('matches', [])))
    if matches:
        print(f"Found entry for section '{section_num}'.")
    else:
        print(f"No entry for section '{section_num}'.")
    return matches


def check_pdf_in_pinecone(pdf_path: str) -> bool:
    """
    Extracts metadata from a PDF and checks Pinecone for existence.
    """
    metadata = extract_metadata_dict(pdf_path)
    return check_in_pinecone(metadata)


def main():
    """
    Parse arguments, generate metadata, and check Pinecone for existing entries.
    """
    parser = argparse.ArgumentParser(
        description="Check if a PDF section exists in a Pinecone index."
    )
    parser.add_argument("pdf_path", help="Path to the PDF file.")
    args = parser.parse_args()

    if not os.path.isfile(args.pdf_path):
        print(f"Error: file not found: {args.pdf_path}")
        sys.exit(1)

    metadata = extract_metadata_dict(args.pdf_path)
    found = check_in_pinecone(metadata)

    status = "ALREADY in Pinecone" if found else "NOT found in Pinecone"
    proj = metadata.get('Project_Number', 'N/A')
    sect = metadata.get('Section_Number', 'N/A')
    print(f"\nConclusion: '{args.pdf_path}' (Project: {proj}, Section: {sect}) -> {status}.")
    sys.exit(0 if found else 11)


if __name__ == "__main__":
    main()
