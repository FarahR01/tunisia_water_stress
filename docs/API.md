# API Documentation

## Overview

Tunisia Water Stress ML provides a RESTful API built with **FastAPI** for making water stress predictions. The API is fully documented with interactive Swagger/OpenAPI documentation.

---

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r api_requirements.txt

# Run API server
python api/main.py
# or
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# View API docs
# Navigate to: http://localhost:8000/docs (Swagger UI)
#             http://localhost:8000/redoc (ReDoc alternative)
```

### Docker

```bash
# Build and run
docker-compose up -d

# API available at: http://localhost:8000
```

---

## Authentication

Currently, the API has **no authentication** (open access). For production, add:
- API keys
- JWT tokens
- OAuth2

See `api/config.py` for configuration options.

---

## Endpoints

### Health Check

**Endpoint:** `GET /health`

**Purpose:** Verify API is running

**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "timestamp": "2024-02-22T10:30:00Z"
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

---

### Single Prediction

**Endpoint:** `POST /v1/predict`

**Purpose:** Get a prediction for one or more feature sets

**Request Body:**
```json
{
  "features": {
    "Population_growth_annual_pct": 1.5,
    "Access_to_improved_water_source_pct": 95.2,
    "year": 2023
  },
  "model_name": "RandomForest"
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `features` | Dict[str, float] | Yes | - | Feature values for prediction |
| `model_name` | string | No | "RandomForest" | Which model to use: "LinearRegression", "Ridge", "Lasso", "DecisionTree", "RandomForest" |

**Response:**
```json
{
  "prediction": 34.7,
  "model_name": "RandomForest",
  "confidence": 0.92,
  "timestamp": "2024-02-22T10:30:00Z"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | float | Predicted water stress value (%) |
| `model_name` | string | Model used for prediction |
| `confidence` | float | Model confidence (0-1) |
| `timestamp` | string | Prediction timestamp (ISO 8601) |

**Example:**
```bash
curl -X POST "http://localhost:8000/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Population_growth_annual_pct": 1.5,
      "Access_to_improved_water_source_pct": 95.2,
      "year": 2023
    },
    "model_name": "RandomForest"
  }'
```

**Error Responses:**

| Status | Code | Message |
|--------|------|---------|
| 422 | `VALIDATION_ERROR` | Missing or invalid fields |
| 404 | `MODEL_NOT_FOUND` | Specified model doesn't exist |
| 500 | `PREDICTION_ERROR` | Model inference failed |

---

### Batch Predictions

**Endpoint:** `POST /v1/batch-predict`

**Purpose:** Get predictions for multiple samples

**Request Body:**
```json
{
  "data": [
    {
      "Population_growth_annual_pct": 1.5,
      "Access_to_improved_water_source_pct": 95.2,
      "year": 2023
    },
    {
      "Population_growth_annual_pct": 1.6,
      "Access_to_improved_water_source_pct": 95.5,
      "year": 2024
    }
  ],
  "model_name": "RandomForest"
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `data` | List[Dict] | Yes | - | List of feature dictionaries |
| `model_name` | string | No | "RandomForest" | Which model to use |

**Response:**
```json
{
  "predictions": [34.7, 35.2],
  "model_name": "RandomForest",
  "count": 2,
  "processing_time_ms": 45,
  "timestamp": "2024-02-22T10:30:00Z"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `predictions` | List[float] | Predicted water stress values |
| `model_name` | string | Model used |
| `count` | int | Number of predictions |
| `processing_time_ms` | float | Inference time in milliseconds |
| `timestamp` | string | Prediction timestamp |

**Limits:**
- Maximum batch size: 1000 predictions
- Returns 413 if exceeded

**Example:**
```bash
curl -X POST "http://localhost:8000/v1/batch-predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"Population_growth_annual_pct": 1.5, "year": 2023},
      {"Population_growth_annual_pct": 1.6, "year": 2024}
    ],
    "model_name": "RandomForest"
  }'
```

---

### Get Model Info

**Endpoint:** `GET /v1/models`

**Purpose:** List available models and their info

**Response:**
```json
{
  "models": [
    {
      "name": "RandomForest",
      "type": "ensemble",
      "r2_score": 0.92,
      "mae": 2.3,
      "status": "active"
    },
    {
      "name": "Ridge",
      "type": "linear",
      "r2_score": 0.88,
      "mae": 3.1,
      "status": "active"
    }
  ],
  "default_model": "RandomForest"
}
```

**Example:**
```bash
curl http://localhost:8000/v1/models
```

---

## Feature Names & Validation

### Available Features

Check `api/config.py` for full feature list. Common features:

```python
FEATURE_NAMES = [
    "year",
    "Population_growth_annual_pct",
    "Access_to_improved_water_source_pct",
    "Access_to_improved_sanitation_pct",
    "Agricultural_land_pct_of_land_area",
    "Arable_land_pct_of_land_area",
    # ... more features
]
```

### Feature Constraints

- **Numeric features**: Expected to be in reasonable ranges
- **Missing features**: Function will interpolate if possible
- **Out-of-range values**: May produce unreliable predictions (no hard constraint)

**Example valid request:**
```json
{
  "features": {
    "year": 2023,
    "Population_growth_annual_pct": 1.5,
    "Access_to_improved_water_source_pct": 95.2
  }
}
```

---

## Error Handling

### Common Errors

**422 Unprocessable Entity**
```json
{
  "detail": [
    {
      "loc": ["body", "features"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

Solution: Ensure all required fields are included.

**400 Bad Request**
```json
{
  "detail": "Invalid model_name. Available: RandomForest, Ridge, Lasso"
}
```

Solution: Use one of the available model names.

**404 Not Found**
```json
{
  "detail": "Model RandomForest not found"
}
```

Solution: Check model names via `/v1/models` endpoint.

**500 Internal Server Error**
```json
{
  "detail": "Error during prediction: ..."
}
```

Solution: Check server logs; contact maintainers if persistent.

---

## Usage Examples

### Python

```python
import requests

BASE_URL = "http://localhost:8000"

# Single prediction
response = requests.post(
    f"{BASE_URL}/v1/predict",
    json={
        "features": {
            "year": 2023,
            "Population_growth_annual_pct": 1.5,
            "Access_to_improved_water_source_pct": 95.2
        },
        "model_name": "RandomForest"
    }
)
print(response.json())
# Output: {'prediction': 34.7, 'model_name': 'RandomForest', 'confidence': 0.92, ...}

# Batch predictions
response = requests.post(
    f"{BASE_URL}/v1/batch-predict",
    json={
        "data": [
            {"year": 2023, "Population_growth_annual_pct": 1.5},
            {"year": 2024, "Population_growth_annual_pct": 1.6}
        ],
        "model_name": "Ridge"
    }
)
print(response.json())
# Output: {'predictions': [34.7, 35.2], 'count': 2, 'processing_time_ms': 45, ...}
```

### JavaScript/Node.js

```javascript
const BASE_URL = "http://localhost:8000";

// Single prediction
fetch(`${BASE_URL}/v1/predict`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    features: {
      year: 2023,
      Population_growth_annual_pct: 1.5,
      Access_to_improved_water_source_pct: 95.2
    },
    model_name: "RandomForest"
  })
})
.then(res => res.json())
.then(data => console.log(data));
```

### cURL

```bash
# Single prediction
curl -X POST "http://localhost:8000/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "year": 2023,
      "Population_growth_annual_pct": 1.5
    }
  }'

# List available models
curl "http://localhost:8000/v1/models"

# Health check
curl "http://localhost:8000/health"
```

---

## Performance & Benchmarks

### Latency

| Operation | Time |
|-----------|------|
| Single prediction | 10-50ms |
| Batch 100 predictions | 100-500ms |
| Model load (startup) | 100-500ms |

### Throughput

- **Single instance** (1 process): ~100-500 req/sec
- **Gunicorn** (4 workers): ~400-2000 req/sec
- **Docker Compose**: Depends on resource allocation

### Memory Usage

| Component | Memory |
|-----------|--------|
| API + dependencies | 200-300 MB |
| Models in memory | 50-200 MB |
| Single request | < 10 MB |
| **Total container** | ~500 MB |

---

## Deployment

### Production Checklist

- [ ] Enable authentication (API keys or JWT)
- [ ] Add rate limiting
- [ ] Set up error tracking (Sentry)
- [ ] Enable structured logging (ELK stack)
- [ ] Add HTTPS/TLS
- [ ] Configure CORS if serving multiple origins
- [ ] Set up monitoring and alerts
- [ ] Document SLA and maintenance windows
- [ ] Create runbooks for common issues
- [ ] Set up automated backups of models

### Scaling

**Horizontal scaling:**
```bash
# Run multiple API instances behind Nginx
docker-compose up -d --scale api=4

# Nginx automatically load balances
```

**Vertical scaling:**
```yaml
# In docker-compose.yml
services:
  api:
    mem_limit: 2g  # Increase memory
    cpus: 2.0      # Allocate more CPU cores
```

---

## Configuration

### Environment Variables

Set in `.env` or `docker-compose.yml`:

```bash
# API Configuration
API_VERSION=1.0.0
DEFAULT_MODEL=RandomForest
HOST=0.0.0.0
PORT=8000

# Model Configuration
MODELS_DIR=models/
MODEL_TIMEOUT=30

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json  # or "text"

# CORS
CORS_ENABLED=false
ALLOWED_ORIGINS=["http://localhost:3000"]
```

### Model Configuration

Edit `api/config.py`:

```python
# Available models
MODEL_PATHS = {
    "LinearRegression": "models/LinearRegression.joblib",
    "Ridge": "models/Ridge.joblib",
    "Lasso": "models/Lasso.joblib",
    "DecisionTree": "models/DecisionTree.joblib",
    "RandomForest": "models/RandomForest.joblib",
}

# Default model for predictions
DEFAULT_MODEL = "RandomForest"

# Feature names (in order)
FEATURE_NAMES = [...]

# Maximum batch size
MAX_BATCH_SIZE = 1000
```

---

## Troubleshooting

### API won't start

**Error:** `Address already in use`

**Solution:**
```bash
# Find process using port 8000
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill and restart
kill -9 <PID>
uvicorn api.main:app --port 8001  # Use different port
```

### Model not loading

**Error:** `Model RandomForest not found`

**Solution:**
```bash
# Check models exist
ls models/RandomForest.joblib

# Check config points to correct path
grep MODELS_DIR api/config.py

# Retrain models
python src/train.py --models_dir models/
```

### High latency

**Symptoms:** Predictions taking >1 second

**Solutions:**
1. Check server resources: `docker stats`
2. Reduce batch size (if using batch endpoint)
3. Upgrade hardware or use multiple instances
4. Profile with: `python -m cProfile api/main.py`

### Type errors in requests

**Error:** `Validation error: value is not a valid number`

**Solution:**
- Ensure all feature values are numbers (not strings)
- Check feature names match exactly (case-sensitive)
- Verify required fields are included

---

## Advanced Topics

### Request Logging

Enable detailed request logging in `api/logging_config.py`:

```python
logging.getLogger("uvicorn.access").setLevel(logging.DEBUG)
```

View logs:
```bash
docker logs <container_id>
tail -f logs/api.log
```

### Model Serving Strategies

**Current (Single Model):**
- One active model at a time
- Simple, fast inference

**Future (Ensemble):**
- Combine multiple models
- Average or weighted predictions
- Better robustness

**Future (Canary Deployment):**
- Route 90% to old model, 10% to new
- Monitor metrics before full rollout

### Caching Predictions

Add Redis caching for repeated predictions:

```python
# In api/model_service.py
import redis
cache = redis.Redis(host='cache', port=6379)

def predict(features, model_name):
    cache_key = f"{model_name}:{hash(features)}"
    if cached := cache.get(cache_key):
        return json.loads(cached)
    
    prediction = model.predict(features)
    cache.setex(cache_key, 3600, json.dumps(prediction))
    return prediction
```

---

## Documentation

- [Project Handbook](../docs/PROJECT_HANDBOOK.md) - Full development guide
- [Architecture](../docs/ARCHITECTURE.md) - System design
- [README](../README.md) - Project overview

---

**Last Updated:** February 2026  
**API Version:** 1.0.0
