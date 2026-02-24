# Use Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dashboard files
COPY dashboard/ /app/
COPY data/ /data/
COPY models_tuned/ /models/
COPY artifacts/ /artifacts/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
