#!/bin/bash
# Run Tunisia Water Stress ML API locally (Linux/Mac)

echo "Starting Tunisia Water Stress ML API..."
echo

# Check if venv exists
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Warning: Virtual environment not found, using system Python"
fi

# Install API requirements if needed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing API dependencies..."
    pip install -r api_requirements.txt
fi

# Start the API server
echo
echo "API starting at http://localhost:8000"
echo "Documentation available at http://localhost:8000/docs"
echo

uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
