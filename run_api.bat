@echo off
REM Run Tunisia Water Stress ML API locally (Windows)

echo Starting Tunisia Water Stress ML API...
echo.

REM Check if venv exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found, using system Python
)

REM Install API requirements if needed
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo Installing API dependencies...
    pip install -r api_requirements.txt
)

REM Start the API server
echo.
echo API starting at http://localhost:8000
echo Documentation available at http://localhost:8000/docs
echo.

uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
