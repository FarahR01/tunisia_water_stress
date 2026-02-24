# Tunisia Water Stress ML API
# Multi-stage build for smaller production image

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY api_requirements.txt .
RUN pip install --no-cache-dir -r api_requirements.txt

# Production stage
FROM python:3.11-slim as production

WORKDIR /app

# Copy Python packages from builder - packages are installed globally
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application code
COPY api/ ./api/
COPY models_tuned/ ./models_tuned/
COPY data/processed/ ./data/processed/

# Create non-root user for security
RUN useradd -m -u 1000 apiuser && \
    chown -R apiuser:apiuser /app
USER apiuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/api/v1/health', timeout=5).raise_for_status()" || exit 1

# Run the application - using python -m to avoid permission issues
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
