"""
Configuration module for Submittal Factory backend
Centralizes environment variable handling and production settings
"""

import os
import json
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file from the backend directory
backend_dir = Path(__file__).parent
dotenv_path = backend_dir / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
else:
    # Try loading from parent directory as fallback
    parent_dotenv = backend_dir.parent / '.env'
    if parent_dotenv.exists():
        load_dotenv(dotenv_path=parent_dotenv)


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Keys
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    serpapi_api_key: str = Field(default="", alias="SERPAPI_API_KEY")
    pinecone_api_key: str = Field(default="", alias="PINECONE_API_KEY")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    debug: bool = Field(default=False, alias="DEBUG")
    reload: bool = Field(default=False, alias="RELOAD")
    
    # CORS Settings - Default to allowing common development origins
    cors_origins: list[str] = Field(default=["*"], alias="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(default=True, alias="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: list[str] = Field(default=["*"], alias="CORS_ALLOW_METHODS")
    cors_allow_headers: list[str] = Field(default=["*"], alias="CORS_ALLOW_HEADERS")
    
    # File Upload Settings
    max_upload_size_mb: int = Field(default=50, alias="MAX_UPLOAD_SIZE_MB")

    # --- NEW: Database URL for SQLAlchemy / psycopg2
    database_url: str = Field(..., alias="DATABASE_URL")

    # --- NEW: JWT Auth settings
    jwt_secret: str = Field(..., alias="JWT_SECRET")
    jwt_expiration_hours: int = Field(default=1, alias="JWT_EXPIRATION_HOURS")

    jwt_issuer: str = Field(..., alias="JWT_ISSUER")
    jwt_audience: str = Field(..., alias="JWT_AUDIENCE")

    @field_validator('cors_origins', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from various string formats"""
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            # Try JSON format first (e.g., '["url1","url2"]')
            if v.startswith('[') and v.endswith(']'):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            # Try comma-separated format (e.g., 'url1,url2,url3')
            elif ',' in v:
                return [url.strip() for url in v.split(',') if url.strip()]
            # Single URL
            else:
                return [v.strip()] if v.strip() else ["*"]
        return ["*"]
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"  # Ignore extra fields from environment
    }


# Create global settings instance
settings = Settings()

# Validate required settings
def validate_required_settings():
    """Validate that required environment variables are set"""
    errors = []
    
    if not settings.google_api_key:
        errors.append("GOOGLE_API_KEY is required. Get one from: https://aistudio.google.com/app/apikey")
    
    if not settings.serpapi_api_key:
        # SerpAPI is optional but warn if missing
        print("Warning: SERPAPI_API_KEY not set. Search functionality will be limited.")
        print("Get a key from: https://serpapi.com/")
        # Set a dummy key for basic functionality
        settings.serpapi_api_key = "dummy_key_for_basic_functionality"

    if not settings.database_url:
        errors.append("DATABASE_URL is required. e.g. postgres://user:pass@host:port/db?sslmode=require")
    
    if not settings.jwt_secret:
        errors.append("JWT_SECRET is required. Choose a long, random string.")
    
    if errors:
        raise ValueError(
            "Missing required configuration:\n" +
            "\n".join(f"- {error}" for error in errors)
        )

# Auto-validate on import (can be disabled for testing)
if os.getenv("SKIP_CONFIG_VALIDATION") != "true":
    validate_required_settings()
