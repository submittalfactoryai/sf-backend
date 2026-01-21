FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Build-time deps (builder image size doesn't matter much)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ build-essential pkg-config \
    libffi-dev libssl-dev \
    libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b \
    libgdk-pixbuf-2.0-0 libgdk-pixbuf-xlib-2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------- runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

# Runtime libs (Debian trixie names)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2 \
    libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b \
    libgdk-pixbuf-2.0-0 libgdk-pixbuf-xlib-2.0-0 \
    shared-mime-info \
    fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy installed packages/binaries from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p /app/uploads /app/temp && \
    chown -R appuser:appuser /app

USER appuser

# Health check without third-party deps (no 'requests' needed)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=10).read()" || exit 1

EXPOSE 8000
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
