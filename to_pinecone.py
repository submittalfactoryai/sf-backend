import json
import os
import re
import argparse
import subprocess  # Added for calling metadata.py
import logging  # Add logging import
from pinecone import Pinecone, ServerlessSpec, PodSpec  # Updated import for pinecone-client v3+
from google import genai
from google.genai import types
from tqdm.auto import tqdm  # Progress bar
import time  # To handle potential rate limits
from dotenv import load_dotenv

# Import functions from local modules
try:
    from to_text import extract_text_from_pdf, split_text_by_part  # Import split_text_by_part
    from metadata import main as extract_metadata  # Import metadata extraction function
except ImportError:
    from .to_text import extract_text_from_pdf, split_text_by_part  # Import split_text_by_part
    try:
        from backend.metadata import main as extract_metadata  # Import metadata extraction function
    except ImportError:
        from .metadata import main as extract_metadata  # Import metadata extraction function

load_dotenv()  # Add this near the top of your script

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---

# Pinecone Configuration
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY") or "YOUR_PINECONE_API_KEY"  # Replace or set env var
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT") or "YOUR_PINECONE_ENVIRONMENT"  # Replace or set env var (Cloud region for PodSpec)

# Vertex AI Configuration
GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or "submittalfactoryai"
GOOGLE_CLOUD_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION") or "us-central1"
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or ""

# Set credentials path for Google Cloud SDK
if GOOGLE_APPLICATION_CREDENTIALS:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

# Specify the Gemini Embedding model
# NOTE: Ensure your Pinecone index dimension matches this model (typically 768 for embedding-001/gemini models)
EMBEDDING_MODEL_NAME = "text-embedding-004"  # Updated model name for Vertex AI

# Other Configuration
BATCH_SIZE = 100  # Process records in batches for upserting (Pinecone limit)
EMBEDDING_BATCH_SIZE = 100  # How many texts to embed in one API call to Google (max 100)

# Initialize Vertex AI client
def get_vertex_client():
    """Get or create Vertex AI client."""
    return genai.Client(
        vertexai=True,
        project=GOOGLE_CLOUD_PROJECT,
        location=GOOGLE_CLOUD_LOCATION
    )

# --- Helper Function ---
def generate_safe_id(text, max_length=512):
    """Generates a Pinecone-safe ID from text."""
    # Remove non-alphanumeric characters (except hyphen and underscore)
    safe_text = re.sub(r'[^a-zA-Z0-9_-]', '_', text)
    # Truncate if too long
    return safe_text[:max_length]

def chunk_text(text, max_chunk_size=2000, overlap=200):
    """
    Splits the text into overlapping chunks for embedding and retrieval.
    Args:
        text (str): The full extracted text.
        max_chunk_size (int): Maximum number of characters per chunk.
        overlap (int): Number of overlapping characters between chunks.
    Returns:
        List[str]: List of text chunks.
    """
    if not text:
        return []
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + max_chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == text_length:
            break
        start = end - overlap  # overlap for context
    return chunks

# --- Main Script ---
def main(pdf_path):
    # 1. Validate Configuration
    if not PINECONE_API_KEY or PINECONE_API_KEY == "YOUR_PINECONE_API_KEY":
        logging.error("Error: PINECONE_API_KEY is not set.")
        return
    if not os.path.exists(pdf_path):
        logging.error(f"Error: PDF file not found at {pdf_path}")
        return

    # --- Derive base name for output files ---
    pdf_dir = os.path.dirname(pdf_path)
    pdf_basename = os.path.basename(pdf_path)
    base_name = os.path.splitext(pdf_basename)[0]
    temp_text_path = os.path.join(pdf_dir, f"{base_name}_to_text.txt")
    temp_metadata_path = os.path.join(pdf_dir, f"{base_name}_metadata_temp.txt")

    # 2. Extract metadata using metadata.py and save as temp text file
    logging.info(f"Extracting metadata from {pdf_path} using metadata.py...")
    extract_metadata(pdf_path, output_filename=temp_metadata_path)
    if not os.path.exists(temp_metadata_path):
        logging.error(f"Error: Metadata extraction failed for {pdf_path}")
        return
    with open(temp_metadata_path, 'r', encoding='utf-8') as f:
        pdf_metadata = json.load(f)
    if isinstance(pdf_metadata, list):
        if len(pdf_metadata) > 0 and isinstance(pdf_metadata[0], dict):
            pdf_metadata = pdf_metadata[0]
        else:
            logging.error(f"Metadata in {temp_metadata_path} is a list but does not contain a valid dict.")
            return
    logging.info(f"Extracted metadata saved to: {temp_metadata_path}")

    # Set Pinecone index and namespace from metadata
    pinecone_index_name = str(pdf_metadata.get("Project_Number", "default")).strip().lower().replace('_', '-').replace(' ', '-').replace('.', '-')
    # Namespace: Section_Number, replace spaces with hyphens, slashes with hyphens, dots with hyphens (for Pinecone compatibility)
    pinecone_namespace = str(pdf_metadata.get("Section_Number", "default")).strip().replace(' ', '-').replace('/', '-').replace('.', '-')

    # --- Clean metadata for Pinecone (remove None/null values, ensure valid types) ---
    def clean_metadata(md):
        return {k: v for k, v in md.items() if v is not None and (isinstance(v, (str, int, float, bool)) or (isinstance(v, list) and all(isinstance(i, str) for i in v)))}

    # 3. Extract text using to_text.py logic
    logging.info(f"Extracting text from {pdf_path} using extract_text_from_pdf...")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        logging.error(f"Error: Could not extract text from {pdf_path}")
        return

    # Use only the second part after splitting by PART X - ...
    parts = split_text_by_part(text)
    if len(parts) > 1:
        text_to_upload = parts[1]
    else:
        logging.warning("split_text_by_part did not find a second part; uploading full text instead.")
        text_to_upload = text

    with open(temp_text_path, 'w', encoding='utf-8') as f:
        f.write(text_to_upload)
    logging.info(f"Extracted text saved to: {temp_text_path}")

    # --- Chunk the extracted text for embedding/retrieval ---
    chunks = chunk_text(text_to_upload, max_chunk_size=2000, overlap=200)
    logging.info(f"Text split into {len(chunks)} chunks for embedding.")

    # --- Initialize Vertex AI Client ---
    logging.info(f"Initializing Vertex AI client for embeddings with model: {EMBEDDING_MODEL_NAME}...")
    try:
        vertex_client = get_vertex_client()
        expected_dimension = 768
        logging.info(f"Expected embedding dimension: {expected_dimension}")
    except Exception as e:
        logging.error(f"Error initializing Vertex AI client: {e}")
        return

    # --- Initialize Pinecone Connection ---
    logging.info(f"Initializing Pinecone connection...")
    try:
        pinecone = Pinecone(api_key=PINECONE_API_KEY)
        existing_indexes = pinecone.list_indexes()
        existing_index_names = [index.name for index in existing_indexes.indexes]

        if pinecone_index_name not in existing_index_names:
            logging.info(f"Index '{pinecone_index_name}' does not exist. Creating it now...")
            pinecone.create_index(
                name=pinecone_index_name,
                dimension=expected_dimension,
                metric='cosine',
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logging.info(f"Index '{pinecone_index_name}' created. Waiting for initialization...")
            time.sleep(60)

        index = pinecone.Index(pinecone_index_name)
        logging.info("Pinecone connection established.")
        index_stats = index.describe_index_stats()
        pinecone_dimension = index_stats.dimension
        if pinecone_dimension != expected_dimension:
            logging.critical(f"CRITICAL WARNING: Pinecone index dimension ({pinecone_dimension}) does not match expected model dimension ({expected_dimension})!")
    except Exception as e:
        logging.error(f"Error connecting to Pinecone: {e}")
        return

    # --- Prepare Data for Embedding ---
    logging.info("Preparing data for embedding...")
    vectors_to_upsert = []
    items_to_embed = []
    for i, chunk in enumerate(chunks):
        unique_id = generate_safe_id(f"{base_name}_chunk_{i}")
        chunk_metadata = clean_metadata(pdf_metadata.copy())
        chunk_metadata.update({
            "file_name": pdf_basename,
            "chunk_index": i,
            "text_snippet": chunk[:200]
        })
        chunk_metadata = clean_metadata(chunk_metadata)
        items_to_embed.append({
            "id": unique_id,
            "text": chunk,
            "metadata": chunk_metadata
        })

    # --- Embed data using Google AI API ---
    logging.info(f"Generating embeddings for {len(items_to_embed)} items...")
    all_embeddings = []
    try:
        response = vertex_client.models.embed_content(
            model=EMBEDDING_MODEL_NAME,
            contents=[item['text'] for item in items_to_embed],
        )
        if hasattr(response, 'embeddings') and len(response.embeddings) == len(items_to_embed):
            all_embeddings.extend([emb.values for emb in response.embeddings])
        else:
            logging.warning(f"Warning: Mismatch in embedding count or missing 'embeddings'. Expected {len(items_to_embed)}")
            all_embeddings.extend([None] * len(items_to_embed))
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        all_embeddings.extend([None] * len(items_to_embed))

    if len(all_embeddings) != len(items_to_embed):
        logging.error(f"Error: Number of generated embeddings ({len(all_embeddings)}) does not match number of items ({len(items_to_embed)}). Aborting upsert.")
        return

    vectors_to_upsert = []
    for item, embedding in zip(items_to_embed, all_embeddings):
        if embedding is not None:
            vectors_to_upsert.append(
                (item['id'], embedding, item['metadata'])
            )
        else:
            logging.warning(f"Skipping item ID {item['id']} due to embedding error.")

    # --- Upsert to Pinecone ---
    if vectors_to_upsert:
        logging.info(f"Upserting {len(vectors_to_upsert)} vectors to Pinecone index '{pinecone_index_name}' (namespace: '{pinecone_namespace}')...")
        try:
            index.upsert(vectors=vectors_to_upsert, namespace=pinecone_namespace)
            logging.info(f"Successfully upserted {len(vectors_to_upsert)} vectors to namespace '{pinecone_namespace}'.")
        except Exception as e:
            logging.error(f"Error during Pinecone upsert to namespace '{pinecone_namespace}': {e}")
    else:
        logging.warning("No vectors prepared for upserting (possibly due to embedding errors or no relevant content found).")

def upload_pdf_to_pinecone(pdf_path: str):
    """
    Uploads the PDF content to Pinecone using extracted metadata and text.
    This is the main function for programmatic use.
    """
    main(pdf_path)

# --- Run Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text and upload PDF content to Pinecone.")
    parser.add_argument("pdf_path", help="Path to the input PDF file (e.g., ai/SEC0.pdf)")
    args = parser.parse_args()

    main(args.pdf_path)

