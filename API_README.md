# Tunisia Water Stress ML API

A RESTful API for water stress predictions in Tunisia using machine learning models.

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r api_requirements.txt

# Run the API server
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Or use the convenience script (Windows)
run_api.bat

# Or (Linux/Mac)
./run_api.sh
```

The API will be available at `http://localhost:8000`

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build manually
docker build -t tunisia-water-stress-api .
docker run -p 8000:8000 tunisia-water-stress-api
```

## API Documentation

Interactive documentation is available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpoints

### Health Check
```
GET /health
```
Returns API status and number of loaded models.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": 5
}
```

### Get Models Info
```
GET /models
```
Returns available models, their metrics, and expected features.

**Response:**
```json
{
  "available_models": ["LinearRegression", "DecisionTree", "RandomForest", "Ridge", "Lasso"],
  "default_model": "Lasso",
  "metrics": [...],
  "feature_columns": [...]
}
```

### Make Prediction
```
POST /predict
```
Make a single water stress prediction.

**Request Body:**
```json
{
  "year": 2025,
  "water_productivity": 8.5,
  "freshwater_withdrawals_agriculture": 75.0,
  "freshwater_withdrawals_industry": 5.0,
  "renewable_freshwater_resources": 4.195,
  "model_name": "Lasso"
}
```

Note: All fields except `year` are optional. Missing features will be extrapolated from historical data.

**Response:**
```json
{
  "year": 2025,
  "predicted_water_stress": 59.2,
  "model_used": "Lasso",
  "input_features": {...},
  "confidence_note": "This prediction is based on historical patterns..."
}
```

### Scenario Prediction
```
POST /predict/scenario
```
Generate a scenario-based prediction with trend extrapolation.

**Request Body:**
```json
{
  "target_year": 2030,
  "trend_method": "linear",
  "model_name": "Lasso"
}
```

**Trend Methods:**
- `linear`: Linear extrapolation from recent trends (default)
- `exponential`: Exponential growth/decay
- `last_value`: Use most recent values
- `average`: Use historical averages

**Response:**
```json
{
  "target_year": 2030,
  "predicted_water_stress": 56.94,
  "model_used": "Lasso",
  "trend_method": "linear",
  "extrapolated_features": {...},
  "interpretation": "By 2030, Tunisia is predicted to have medium-high water stress..."
}
```

### Batch Predictions
```
POST /predict/batch
```
Make multiple predictions in a single request.

**Request Body:**
```json
{
  "predictions": [
    {"year": 2025},
    {"year": 2026},
    {"year": 2027}
  ]
}
```

### Year Range Predictions
```
GET /predict/years/{start_year}/{end_year}
```
Get predictions for a range of years.

**Example:**
```
GET /predict/years/2025/2035
```

## Usage Examples

### Python
```python
import httpx

# Health check
response = httpx.get("http://localhost:8000/health")
print(response.json())

# Single prediction
response = httpx.post(
    "http://localhost:8000/predict",
    json={"year": 2030}
)
print(response.json())

# Scenario prediction
response = httpx.post(
    "http://localhost:8000/predict/scenario",
    json={"target_year": 2030, "trend_method": "linear"}
)
print(response.json())
```

### cURL
```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"year": 2030}'

# Scenario prediction
curl -X POST http://localhost:8000/predict/scenario \
  -H "Content-Type: application/json" \
  -d '{"target_year": 2030, "trend_method": "linear"}'
```

### JavaScript/Fetch
```javascript
// Single prediction
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ year: 2030 })
});
const data = await response.json();
console.log(data);
```

## Available Models

| Model | MAE | RMSE | R² | Notes |
|-------|-----|------|----|----|
| **Lasso** | 4.08 | 4.60 | 0.30 | Best performer (default) |
| Ridge | 17.66 | 18.75 | -10.70 | - |
| RandomForest | 17.41 | 18.32 | -10.16 | - |
| DecisionTree | 18.76 | 19.70 | -11.90 | - |
| LinearRegression | 226.24 | 247.84 | -2042.13 | - |

## Features Used

The models use the following features:
- `year`: Year for prediction
- `Water productivity`: Constant 2015 US$ GDP per cubic meter
- `Annual freshwater withdrawals, agriculture`: % of total
- `Annual freshwater withdrawals, industry`: % of total
- `Renewable internal freshwater resources`: Billion cubic meters

## Production Deployment

### With Docker Compose (Recommended)

```bash
# Start with nginx reverse proxy
docker-compose --profile production up -d
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | Server port |
| `HOST` | 0.0.0.0 | Server host |
| `WORKERS` | 1 | Number of workers (for gunicorn) |

### Kubernetes

A basic Kubernetes deployment can use the Dockerfile with:
- Deployment with 2+ replicas
- Service (ClusterIP or LoadBalancer)
- Horizontal Pod Autoscaler

## Project Structure

```
tunisia_water_stress_ml/
├── api/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── config.py         # Configuration settings
│   ├── schemas.py        # Pydantic models
│   └── model_service.py  # Model loading & prediction
├── models_tuned/         # Trained models
├── data/processed/       # Historical data
├── Dockerfile
├── docker-compose.yml
├── api_requirements.txt
└── API_README.md         # This file
```

## License

See main project README for license information.
