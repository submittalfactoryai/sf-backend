# Submittal Factory Backend

FastAPI-based backend service for automated construction submittal validation.

## 🏗️ Architecture

The backend is built with:
- **FastAPI** - Modern, fast web framework for building APIs
- **Google Gemini AI** - For intelligent document processing and validation
- **PyMuPDF** - PDF text extraction and processing
- **Pinecone** - Vector database for similarity search (optional)
- **SerpAPI** - Enhanced web search capabilities (optional)

## 📁 Project Structure

```
backend/
├── api_server.py           # Main FastAPI application
├── config.py              # Configuration management
├── start_backend.py        # Backend startup script
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create from .env.example)
├── .env.example            # Environment variables template
├── .dockerignore          # Docker build exclusions
├── Dockerfile             # Container definition
└── modules/
    ├── extract_with_llm.py    # AI-powered data extraction
    ├── validate_with_llm.py   # Specification validation
    ├── searching_pdf.py       # PDF search functionality
    ├── to_text.py             # Text extraction utilities
    ├── clean_text.py          # Text preprocessing
    ├── to_json.py             # JSON structure parsing
    ├── metadata.py            # Document metadata generation
    └── pdf_*.py               # PDF processing utilities
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Virtual environment (recommended)
- Required API keys (see Environment Setup)

### Installation

1. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate     # Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

### Environment Setup

Create a `.env` file in the `backend/` directory from `.env.example`:

```bash
# Required
GOOGLE_API_KEY=your_google_gemini_api_key

# Optional (but recommended)
SERPAPI_API_KEY=your_serpapi_key
PINECONE_API_KEY=your_pinecone_key

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false
RELOAD=false

# CORS Configuration
CORS_ORIGINS=["http://localhost:3000","http://localhost:5173","http://localhost:80"]
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=["GET","POST","PUT","DELETE","OPTIONS"]
CORS_ALLOW_HEADERS=["*"]
```

### Running the Backend

**Development Mode:**
```bash
python start_backend.py
```

**Production Mode:**
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1
```

**Docker (Standalone):**
```bash
# Build the image
docker build -t submittal-backend .

# Run the container
docker run -d \
  --name submittal-backend \
  -p 8000:8000 \
  --env-file .env \
  submittal-backend
```

## 📡 API Endpoints

### Core Endpoints

- `GET /api/health` - Health check endpoint
- `POST /api/extract` - Extract data from uploaded PDFs
- `POST /api/validate-specs` - Validate product specifications
- `POST /api/search-submittals` - Search for product submittals

### File Operations

- `GET /api/proxy-pdf` - Proxy PDF downloads
- `POST /api/download-pdfs` - Batch download PDFs
- `POST /api/add-validated-pdfs` - Add validated PDFs to session

### Utilities

- `POST /api/extract-pds-links` - Extract product data sheet links
- `GET /api/usage-summary` - Get API usage statistics

### Documentation

- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation

## 🔧 Configuration

### Server Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `DEBUG` | `false` | Enable debug mode |
| `RELOAD` | `false` | Enable auto-reload |

### CORS Settings

Configure allowed origins for cross-origin requests:

```bash
CORS_ORIGINS=["http://localhost:3000","http://yourdomain.com"]
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=["GET","POST","PUT","DELETE","OPTIONS"]
CORS_ALLOW_HEADERS=["*"]
```

### Upload Limits

```bash
MAX_UPLOAD_SIZE_MB=50  # Maximum file size in MB
```

## 📊 Monitoring

### Health Checks

```bash
curl http://localhost:8000/api/health
```

Response:
```json
{
  "status": "healthy",
  "message": "Backend server is running",
  "timestamp": "2025-05-31T14:22:09.379069+00:00"
}
```

### Usage Tracking

The backend automatically logs API usage to `usage_log.csv` including:
- Request timestamps
- File sizes and types
- AI model usage and costs
- Token consumption

## 🔒 Security

### API Key Management

- Store API keys in the `backend/.env` file
- Never commit API keys to version control
- Use different keys for development and production

### File Upload Security

- File size limits enforced
- PDF file type validation
- Temporary file cleanup
- Secure file handling

### CORS Security

- Restrict origins to your domains only
- Configure appropriate headers
- Enable credentials only when necessary

## 🐛 Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Ensure you're in the backend directory
cd backend
python start_backend.py
```

**API Key Errors:**
```bash
# Check if .env file exists in backend directory
ls .env

# Verify API key is set
grep GOOGLE_API_KEY .env
```

**Port Already in Use:**
```bash
# Kill existing process
lsof -ti:8000 | xargs kill -9

# Or use different port
echo "PORT=8001" >> .env
python start_backend.py
```

**CORS Issues:**
- Add your frontend URL to `CORS_ORIGINS` in `.env`
- Restart backend after changing CORS settings
- Clear browser cache

### Logs

View backend logs for debugging:
```bash
# Development logs
python start_backend.py

# Production logs (if using systemd)
journalctl -u submittal-backend -f
```

## 🚀 Docker Deployment

### Standalone Docker Deployment

The backend can be deployed independently using Docker:

**Build and Run:**
```bash
# Build the Docker image
docker build -t submittal-backend .

# Run as standalone container
docker run -d \
  --name submittal-backend \
  -p 8000:8000 \
  --env-file .env \
  --restart unless-stopped \
  submittal-backend
```

**Check Status:**
```bash
# View logs
docker logs submittal-backend

# Check health
curl http://localhost:8000/api/health
```

**Environment Variables in Docker:**
```bash
# Create .env file with your configuration
cp .env.example .env
# Edit .env file with your API keys

# Docker will automatically load the .env file
docker run --env-file .env submittal-backend
```

### Production Deployment

```bash
# Use production WSGI server
pip install gunicorn
gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 📚 Development

### Adding New Endpoints

1. Define endpoint in `api_server.py`
2. Add request/response models using Pydantic
3. Implement business logic in separate modules
4. Add appropriate error handling
5. Update API documentation

### Testing

```bash
# Run basic health check
curl http://localhost:8000/api/health

# Test PDF upload
curl -X POST -F "file=@test.pdf" http://localhost:8000/api/extract
```

## 🤝 Contributing

1. Follow existing code style
2. Add type hints for all functions
3. Update documentation for new features
4. Test endpoints before submitting PRs

## 📄 License

This project is licensed under the MIT License. 