# System Architecture

## Overview

Tunisia Water Stress ML follows a modular, layered architecture supporting data ingestion, ML processing, API serving, and monitoring.

---

## Architecture Diagrams

### Data Flow Pipeline

```mermaid
graph TD
    A["World Bank API<br/>(Open Data)"] -->|CSV Extract| B["data/raw/<br/>environment_tun.csv"]
    B --> C["src/data_loader.py<br/>(Load & Pivot)"]
    C -->|Wide Format| D["data/processed/<br/>processed_tunisia.csv"]
    D --> E["src/preprocessing.py<br/>(Clean & Select)"]
    E --> F["src/feature_engineering.py<br/>(Create Features)"]
    F --> G["Temporal Train/Test Split<br/>(1960-2010 / 2011-2024)"]
    G -->|Train Data| H["src/train.py<br/>(Model Training)"]
    H -->|Trained Models| I["models/<br/>*.joblib"]
    G -->|Test Data| J["src/evaluate.py<br/>(Evaluation)"]
    J --> K["models/<br/>metrics.csv & plots"]
    I --> L["api/model_service.py<br/>(Load & Inference)"]
    L --> M["FastAPI REST API<br/>(Predictions)"]
    M --> N["Client Applications"]
    K --> O["Monitoring & Reporting"]
```

### System Architecture

```mermaid
graph TB
    subgraph Client["Client Layer"]
        A["Web UI"]
        B["Mobile App"]
        C["External APIs"]
    end
    
    subgraph API_Layer["API Layer"]
        D["FastAPI App<br/>(api/main.py)"]
        E["Route Handlers<br/>(api/routers/v1.py)"]
        F["Request Validation<br/>(Pydantic Schemas)"]
    end
    
    subgraph Service_Layer["Service Layer"]
        G["Model Service<br/>(api/model_service.py)"]
        H["Prediction Engine"]
        I["Caching Layer"]
    end
    
    subgraph Model_Layer["Model Layer"]
        J["RandomForest<br/>Model"]
        K["Ridge<br/>Model"]
        L["Lasso<br/>Model"]
    end
    
    subgraph Data_Layer["Data Layer"]
        M["Processed Data<br/>(processed_tunisia.csv)"]
        N["Model Artifacts<br/>(.joblib files)"]
        O["Feature Definitions"]
    end
    
    subgraph Infrastructure["Infrastructure"]
        P["Docker Container"]
        Q["Uvicorn Server"]
        R["Nginx Proxy"]
    end
    
    Client -->|HTTP| API_Layer
    API_Layer -->|Requests| E
    E -->|Validates| F
    F -->|Calls| Service_Layer
    Service_Layer -->|Loads| Model_Layer
    Model_Layer -->|Uses| Data_Layer
    Service_Layer -->|Caches| I
    Infrastructure -->|Hosts| API_Layer
    Infrastructure -->|Routes| R
    Infrastructure -->|Serves| Q
```

### ML Pipeline Detail

```mermaid
graph TD
    A["Raw Data<br/>(Long Format)"] -->|1. Load| B["DataFrame<br/>(Pivot to Wide)"]
    B -->|2. Clean| C["Remove<br/>Sparse Cols<br/>Fill Missing"]
    C -->|3. Feature Eng| D["Add Lags<br/>Add Year Col"]
    D -->|4. Select| E["Choose Features<br/>(Drop Collinear)"]
    E -->|5. Split| F["Temporal Split"]
    F -->|Train: 1960-2010| G["X_train, y_train"]
    F -->|Test: 2011-2024| H["X_test, y_test"]
    G -->|6. Train| I["Fit Models<br/>GridSearch<br/>CV"]
    I -->|7. Save| J["Trained Models<br/>.joblib"]
    H -->|8. Evaluate| K["Predictions<br/>Metrics<br/>Plots"]
    J -->|9. Deploy| L["API Inference"]
    K -->|10. Monitor| M["Performance<br/>Tracking"]
```

### Component Interaction

```mermaid
graph LR
    A["src/data_loader.py"] -->|loads| B["CSV Files"]
    C["src/preprocessing.py"] -->|cleans data| D["Training Features"]
    E["src/feature_engineering.py"] -->|transforms| F["ML Features"]
    D --> G["src/train.py<br/>(Orchestrator)"]
    F --> G
    G -->|splits| H["X_train/X_test<br/>y_train/y_test"]
    H -->|trains| I["scikit-learn<br/>Models"]
    I -->|saves| J["models/*.joblib"]
    J -->|loads| K["api/model_service.py"]
    K -->|predicts| L["FastAPI<br/>Endpoints"]
    G -->|evaluates| M["src/evaluate.py"]
    M -->|generates| N["metrics.csv<br/>plots/*.png"]
    N -->|inputs| O["Notebooks<br/>Visualization"]
```

### Error Handling & Validation

```mermaid
graph TD
    A["User Request<br/>(API)"] -->|1. Validate| B{Schema<br/>Valid?}
    B -->|No| C["Return 422<br/>Validation Error"]
    B -->|Yes| D{Model<br/>Loaded?}
    D -->|No| E["Return 503<br/>Service Unavailable"]
    D -->|Yes| F{Features<br/>Present?}
    F -->|No| G["Return 400<br/>Bad Request"]
    F -->|Yes| H["Generate<br/>Prediction"]
    H -->|Success| I["Return 200<br/>Prediction"]
    H -->|Failure| J["Return 500<br/>Internal Error<br/>Log Exception"]
```

---

## Component Responsibilities

### Data Pipeline (`src/`)

| Component | Responsibility | I/O |
|-----------|-----------------|-----|
| `data_loader.py` | Load WB data, pivot to wide format | CSV → DataFrame |
| `preprocessing.py` | Clean, fill, select features | DataFrame → DataFrame |
| `feature_engineering.py` | Create lags, temporal features | DataFrame → DataFrame |
| `train.py` | Orchestrate pipeline, train models | CSV → .joblib |
| `evaluate.py` | Compute metrics, generate plots | DataFrame → metrics.csv, PNG |

### API Layer (`api/`)

| Component | Responsibility | Interface |
|-----------|-----------------|-----------|
| `main.py` | Application entry, lifecycle | FastAPI app |
| `routers/v1.py` | Define HTTP endpoints | POST /v1/predict, /v1/batch-predict |
| `schemas.py` | Request/response validation | Pydantic models |
| `model_service.py` | Load models, run inference | Python API |
| `config.py` | Configuration management | Python config dict |
| `dependencies.py` | Dependency injection | FastAPI dependencies |
| `logging_config.py` | Structured logging | Python logging |

### Testing (`tests/`)

| Test Module | Coverage | Focus |
|------------|----------|-------|
| `test_data_loader.py` | Data loading functions | 9 tests, 100% coverage |
| `test_preprocessing.py` | Data cleaning | 16 tests, 100% coverage |
| `test_feature_engineering.py` | Feature creation | 18 tests, 100% coverage |
| `test_pipeline_integration.py` | End-to-end pipeline | 8 integration tests |
| `test_api.py` | API endpoints | Endpoint testing |
| `test_model_service.py` | Model inference | Prediction accuracy |

---

## Technology Stack by Layer

### Data Layer
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **CSV/joblib**: File storage

### Processing Layer
- **scikit-learn**: ML algorithms, preprocessing
- **scipy**: Statistical functions
- **seaborn/matplotlib**: Visualization

### API Layer
- **FastAPI**: Web framework
- **Pydantic**: Data validation
- **Uvicorn**: ASGI server

### Infrastructure Layer
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **Nginx**: Reverse proxy
- **Uvicorn**: Application server

### Development Layer
- **pytest**: Unit testing
- **black**: Code formatting
- **flake8**: Linting
- **mypy**: Type checking
- **pre-commit**: Git hooks

---

## Deployment Architecture

### Local Development

```
Developer Machine
├── Virtual Environment (venv)
├── Source Code (src/, api/)
├── Pytest (tests/)
├── Jupyter Notebooks (notebooks/)
└── Pre-commit Hooks (.git/hooks)
```

### Docker Container

```
Docker Image (tunisia-water-stress-api:latest)
├── Base Image (python:3.9-slim)
├── Dependencies (requirements.txt)
├── Application (api/)
├── Models (models/*.joblib)
└── Entrypoint (uvicorn api.main:app)
```

### Production Stack

```
Load Balancer (Client) 
        ↓
Nginx (Reverse Proxy)
        ↓
Docker Container (API)
        ↓
FastAPI (Uvicorn)
        ↓
Model Service
        ↓
Scikit-learn Models + Data
```

---

## Data Flow Details

### Training Phase

1. **Load**: CSV → Wide DataFrame (src/data_loader.py)
2. **Clean**: Drop sparse cols, fill missing (src/preprocessing.py)
3. **Transform**: Create lags, add temporal features (src/feature_engineering.py)
4. **Split**: Temporal split (1960-2010 train, 2011-2024 test)
5. **Train**: GridSearchCV, hyperparameter tuning (src/train.py)
6. **Save**: Persist models to .joblib (joblib.dump)
7. **Evaluate**: Compute metrics, plots (src/evaluate.py)

### Inference Phase

1. **Request**: POST /v1/predict with features
2. **Validate**: Pydantic schema validation (api/schemas.py)
3. **Load**: Load model from disk (once at startup)
4. **Preprocess**: Apply same preprocessing as training
5. **Predict**: model.predict(features)
6. **Return**: JSON response with prediction

---

## Scalability Considerations

### Current Design (Single Model, Single Process)

- ✓ Simple, easy to develop and test
- ✓ Suitable for moderate traffic (100s req/sec)
- ✗ No horizontal scaling
- ✗ Single point of failure

### Future Enhancements

1. **Model Ensemble**: Load multiple models, average predictions
2. **Async Processing**: Use FastAPI async/await for concurrent requests
3. **Caching**: Redis for feature/prediction caching
4. **Load Balancing**: Multiple API instances behind Nginx
5. **Message Queue**: Celery/RabbitMQ for async jobs
6. **Monitoring**: Prometheus metrics, Grafana dashboards

---

## Security Architecture

### Input Validation
- Pydantic schemas validate all inputs
- Type hints enable early error detection
- Range checks on feature values

### Data Protection
- Models trained on non-PII data (public WB indicators)
- No sensitive data stored in API
- HTTPS/TLS in production

### Code Quality
- Type hints for runtime safety
- Pre-commit security scanning (bandit)
- Dependency vulnerability scanning

### Deployment Security
- Docker container isolation
- Environment variables for secrets
- Nginx security headers

---

## Monitoring & Observability

### Logging
- Structured logging (api/logging_config.py)
- Request/response tracking
- Error stack traces

### Metrics
- Prediction latency
- Model accuracy (periodic evaluation)
- API response times
- Error rates

### Alerts
- Model performance degradation
- API errors (5xx responses)
- Data quality issues

---

## Integration Points

### External Services
- **World Bank API**: Data source for updates
- **Cloud Storage**: Model versioning (S3, GCS)
- **Monitoring**: Error tracking (Sentry), logging (ELK)

### Internal Services
- **Jupyter Notebooks**: Exploratory analysis
- **CI/CD Pipeline**: Automated testing, deployment
- **Version Control**: Git, GitHub

---

## Performance Characteristics

### Expected Latencies
- Single prediction: 10-50ms (scikit-learn)
- Batch 100 predictions: 100-500ms
- Model loading: 100-500ms (one-time at startup)

### Memory Usage
- Model in memory: ~50-200MB (Random Forest, Ridge, Lasso)
- API + dependencies: ~200-300MB
- Total container: ~500MB

### Throughput
- Single instance: 100-500 requests/sec (single process)
- With Gunicorn (4 workers): 400-2000 requests/sec

---

## Related Documentation

- [Project Handbook](./PROJECT_HANDBOOK.md) - Development guide
- [Design Decisions](./DECISIONS.md) - Why architecture was chosen this way
- [API Reference](./API.md) - Endpoint documentation
- [README](../README.md) - Project overview

---

**Last Updated:** February 2026
