# Design Decisions & Trade-offs

## Table of Contents

1. [Model Selection](#model-selection)
2. [Data Handling](#data-handling)
3. [Architecture](#architecture)
4. [API Design](#api-design)
5. [Testing & Quality](#testing--quality)
6. [Deployment](#deployment)

---

## Model Selection

### Decision: Ridge/Lasso Over XGBoost

**Context**: Initial baseline models (Linear Regression, Decision Tree, Random Forest) showed excellent performance but exhibited signs of data leakage.

**Decision**: Prioritize Ridge and Lasso regression as primary models; make XGBoost optional.

**Reasoning**:

| Aspect | Ridge/Lasso | XGBoost |
|--------|-------------|---------|
| **Interpretability** | High (coefficients show feature impact) | Low (black box) |
| **Data Leakage Risk** | Visible in coefficients | Harder to detect |
| **Computational Cost** | Very low (closed-form or fast gradient) | High (iterative, many hyperparams) |
| **Data Requirements** | Works well with <100 samples | Needs 1000+ for good generalization |
| **Regularization** | Built-in (L1/L2) | Manual tuning |
| **Policy Relevance** | Coefficients = feature importance | Importance scores less intuitive |

**Tunisia Dataset Characteristics**:
- Only ~51 training samples (1960-2010)
- Limited data → simpler models generalize better
- Interpretability critical for policy decisions

**Trade-off**:
- ✓ Better generalization, simpler deployment
- ✗ Potentially lower accuracy on unseen data
- ✗ Less complex pattern capture

**Implementation**:
```python
# Default: Ridge (L2 regularization)
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0, solver='auto')

# Optional: Lasso (L1, feature selection)
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.01)
```

**Rationale**: With limited training data, simpler linear models with regularization offer better bias-variance trade-off. Ridge/Lasso coefficients are directly interpretable for policy makers, whereas XGBoost feature importances are harder to explain.

---

### Decision: Linear Regression as Primary Baseline

**Context**: Need a reference model to detect data quality issues and leakage.

**Decision**: Use unregularized Linear Regression as baseline; flag if R² > 0.95.

**Reasoning**:
- Reveals multicollinearity and leakage immediately
- Fast to train and understand
- Close-form solution (no hyperparameter tuning)
- Coefficients show exact feature impact

**Implementation**:
```python
from sklearn.linear_model import LinearRegression
baseline = LinearRegression()
baseline.fit(X_train, y_train)

# Flag potential leakage if test R² is suspiciously high
r2_test = baseline.score(X_test, y_test)
if r2_test > 0.95:
    warn("Potential data leakage detected! Review features vs target.")
```

---

### Decision: Include Decision Tree & Random Forest

**Context**: Need both linear and non-linear models for comparison.

**Decision**: Include Decision Tree and Random Forest despite higher overfitting risk.

**Reasoning**:
- Demonstrates model comparison
- Tree-based models capture non-linear relationships
- Random Forest provides ensemble effect
- Feature importances offer alternative interpretability

**Safeguards**:
- Hyperparameter tuning (GridSearchCV with cross-validation)
- Max depth limits to prevent overfit
- Feature correlation analysis before training
- Holdout test set evaluation

---

## Data Handling

### Decision: Temporal Train/Test Split (Not Random)

**Context**: Water stress data is time series (1960-2024).

**Decision**: Always use temporal split: Train on 1960-2010, test on 2011-2024.

**Reasoning**:

```
WRONG (Random Split):
┌─────────────────────────────────┐
│ Randomly shuffle years          │
│ Train: [1960, 1975, 2020, ...]  │ ← Train on future data!
│ Test:  [1965, 1990, 2011, ...]  │
└─────────────────────────────────┘
Result: Inflated R² (model sees future), unrealistic evaluation

CORRECT (Temporal Split):
┌──────────────┬─────────────────┐
│ Train        │ Test            │
│ 1960-2010    │ 2011-2024       │
│ (51 years)   │ (14 years)      │
└──────────────┴─────────────────┘
Result: Honest evaluation, realistic future predictions
```

**Implementation**:
```python
train_end_year = 2010
train_mask = df['year'] <= train_end_year
X_train = df[train_mask].drop(['year', 'target'], axis=1)
y_train = df[train_mask]['target']

X_test = df[~train_mask].drop(['year', 'target'], axis=1)
y_test = df[~train_mask]['target']
```

**Justification**: 
- Gold standard in time-series ML (no data leakage possible)
- Policy makers need confidence model works on future data
- Prevents accidentally predicting past with future info

---

### Decision: Forward/Backward Fill for Missing Values (No Interpolation)

**Context**: World Bank data has gaps; some years missing for some indicators.

**Decision**: Use forward fill, then backward fill; drop columns >50% missing.

**Reasoning**:

| Method | Pros | Cons |
|--------|------|------|
| **Forward Fill** | Simple, no assumptions | Propagates old values |
| **Interpolation** | Smoother | Assumes linear trend (wrong) |
| **Drop Missing** | Clean | Loses data |
| **Impute Mean** | Common | Reduces variance |

**Chosen Approach**:
1. Forward fill (carry last known value forward)
2. Backward fill (carry next known value backward)
3. Drop columns with >50% missing
4. Report missing data statistics

**Rationale**:
- World Bank updates data retroactively; forward fill captures "best known value"
- For slow-changing environmental indicators, forward fill is reasonable
- Transparent and auditable
- Drops sparse columns early (better than modeling with >50% missing)

---

### Decision: Drop Features with |correlation| >= 0.99 with Target

**Context**: Detected water-related features in raw data identical to target (data leakage).

**Decision**: Implement automated leakage filter.

**Reasoning**:

Before filtering:
```
"Annual freshwater withdrawals (% of internal resources)" = 0.999 corr with target
"Total freshwater withdrawal" = 0.998 corr with target
```

These are either:
1. The target itself under different names
2. Mathematical transforms of the target
3. Directly causally determined by the target

**Implementation**:
```python
def drop_leakage_features(df: pd.DataFrame, target_col: str, threshold: float = 0.99) -> pd.DataFrame:
    """Drop features too correlated with target (likely leakage)."""
    target = df[target_col]
    correlations = df.drop(target_col, axis=1).corrwith(target).abs()
    leakage_features = correlations[correlations >= threshold].index.tolist()
    if leakage_features:
        print(f"Dropping leakage features: {leakage_features}")
    return df.drop(leakage_features, axis=1)
```

**Impact**: 
- ✓ Prevents unrealistic model performance
- ✗ Reduces feature count
- ✓ Improves transferability to other regions

---

### Decision: Drop Highly Collinear Features (|correlation| >= 0.95)

**Context**: Multiple variations of same indicator cause multicollinearity.

**Decision**: Detect and drop highly correlated features before training.

**Reasoning**:
- Features with cor >= 0.95 provide redundant information
- Leads to coefficient instability (small data changes → large coef changes)
- Linear models particularly sensitive
- Tree models less affected but still suboptimal

**Implementation**:
```python
def drop_collinear_features(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Drop features highly correlated with each other."""
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    drop_cols = [col for col in upper.columns if any(upper[col] >= threshold)]
    return df.drop(drop_cols, axis=1)
```

**Trade-off**:
- ✓ Stable, interpretable models
- ✓ Faster training
- ✗ Loses variations of features (e.g., different measurement methods)

---

## Architecture

### Decision: Separate `src/` (ML) and `api/` (REST)

**Context**: Need both batch training and real-time API endpoints.

**Decision**: Independent modules; API loads trained models but doesn't train.

**Reasoning**:

```
Option 1: Monolithic (Train + API in same process)
├── ✓ Simple to understand
├── ✗ Retraining blocks API
├── ✗ Hard to scale/update independently
└── ✗ Single failure point

Option 2: Separate Modules (Chosen)
├── src/ (ML Pipeline)
│   ├── Train models (scheduled)
│   ├── Save to disk (.joblib)
│   └── Run notebooks for exploration
├── api/ (REST Service)
│   ├── Load saved models at startup
│   ├── Serve predictions (fast)
│   └── No direct training
└── ✓ Decoupled, independently scalable
```

**Implementation**:
```
# Training (runs separately, produces artifact)
$ python src/train.py --models_dir models/
→ Saves: models/RandomForest.joblib, models/Ridge.joblib, models/metrics.csv

# API (uses artifact)
$ python api/main.py
→ Loads: joblib.load('models/RandomForest.joblib')
→ Serves: POST /v1/predict
```

---

### Decision: FastAPI Over Flask

**Context**: Need modern API framework with automatic documentation.

**Decision**: Use FastAPI + Uvicorn.

**Reasoning**:

| Feature | Flask | FastAPI |
|---------|-------|---------|
| **Data Validation** | Manual | Automatic (Pydantic) |
| **Async/Await** | Bolts on | Native |
| **OpenAPI Docs** | Manual | Auto-generated |
| **Type Safety** | None | Full (Python 3.6+ hints) |
| **Performance** | Good | Excellent (Starlette) |
| **Learning Curve** | Easier | Moderate |

**Tunisia Project Fit**:
- Automatic OpenAPI docs valuable for users
- Type hints align with project's mypy setup
- Performance sufficient for ~500 req/sec single instance
- Pydantic validation prevents bad requests

---

### Decision: Modular Router Pattern

**Context**: API needs versioning and multiple endpoints.

**Decision**: Implement router pattern with `routers/v1.py`.

**Reasoning**:
```
Bad: All endpoints in main.py
├── Grows large
├── Hard to maintain
└── Unclear structure

Good: Modular routers
├── api/routers/v1.py → v1 endpoints
├── api/routers/v2.py → v2 endpoints (future)
├── Each router focused
└── Easy to add versions
```

---

## Testing & Quality

### Decision: 100% Coverage on Critical Modules

**Context**: Need high confidence in data pipeline correctness.

**Decision**: Target 100% code coverage on `src/` modules; audit every function with tests.

**Reasoning**:
- Data issues propagate through entire pipeline
- False results worse than no results
- 52 tests (43 unit + 8 integration) across 4 modules
- Every function in `src/` has corresponding test

**Implementation**:
```bash
pytest tests/ --cov=src/ --cov-report=html --cov-report=term
# Target: coverage >= 100% on src modules
```

---

### Decision: Pre-commit Hooks for Code Quality

**Context**: Prevent poor-quality code from being committed.

**Decision**: Install pre-commit with black, flake8, mypy, isort, bandit.

**Reasoning**:
- Errors caught before push (faster feedback)
- Consistent formatting (no "style debates" in PRs)
- Type hints verified before commit
- Security issues flagged early

**Trade-off**:
- ✓ Higher code quality bar
- ✗ Slightly slower commit process (~2-5 seconds)

---

### Decision: Type Hints Throughout

**Context**: Python is dynamically typed; easy to make mistakes.

**Decision**: Add PEP 484 type hints to all `src/` modules.

**Reasoning**:
```python
# Without type hints
def load_and_pivot(data):
    # Is data a str (filename) or DataFrame?
    # What gets returned?
    ...

# With type hints
def load_and_pivot(raw_csv_path: str, processed_path: Optional[str] = None) -> pd.DataFrame:
    """Load CSV and pivot to wide format."""
    ...
```

**Benefits**:
- IDE autocomplete works
- mypy catches bugs at "import" time (not runtime)
- Self-documenting code
- Easier refactoring

---

## API Design

### Decision: RESTful Endpoint Structure

**Context**: Need to serve predictions via HTTP.

**Decision**: POST /v1/predict for single, POST /v1/batch-predict for multiple.

**Reasoning**:
```
Endpoint Pattern:
├── /v1/predict
│   ├── POST single prediction
│   └── Response: {"prediction": 34.5, "model": "RandomForest"}
└── /v1/batch-predict
    ├── POST multiple predictions
    └── Response: {"predictions": [34.5, 35.1, ...]}
```

Rationale:
- Single prediction: Common case, fast response
- Batch prediction: For bulk analysis, amortizes model loading
- Clear versioning (/v1 → /v2 easy to add)

---

### Decision: Pydantic Schemas for Validation

**Context**: Prevent invalid requests from reaching model.

**Decision**: Use Pydantic models for input/output.

**Reasoning**:
```python
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    features: Dict[str, float]
    model_name: Optional[str] = "RandomForest"
    
    # Validation
    class Config:
        min_length = 1  # At least one feature

# Automatic validation:
# - Missing required fields → 422 error
# - Type mismatch → 422 error
# - Out of range → caught by logic
```

**Benefits**:
- OpenAPI schema auto-generated
- Clear API contract
- Type safety (matches source code)

---

## Deployment

### Decision: Docker for Reproducibility

**Context**: API must work identically on different machines.

**Decision**: Containerize API with Docker.

**Reasoning**:
```
Without Docker:
├── "Works on my machine"
├── Different Python versions across developers
├── Dependency conflicts
└── Production differs from dev

With Docker:
├── Same image: dev, staging, production
├── Kernel isolation
├── Reproducible builds
└── Easy horizontal scaling
```

**Dockerfile Strategy**:
- Multi-stage build (reduces image size)
- Slim Python base (security + size)
- Non-root user (security)

---

### Decision: Docker Compose for Local Development

**Context**: API needs models, data, and potentially database.

**Decision**: Use docker-compose locally to mimic production.

**Reasoning**:
- Developers test with containerized API
- Consistent with production setup
- Easy to add services (Redis, DB)

---

## Future Decisions

### To Be Addressed

1. **Model Retraining Strategy**:
   - How often? (Monthly? Quarterly?)
   - Automatic or manual trigger?
   - A/B testing new vs. current model?

2. **Feature Drift Monitoring**:
   - How to detect when World Bank data quality changes?
   - Alert mechanism?

3. **Multi-model Ensembling**:
   - Combine Ridge + RandomForest predictions?
   - Weighted average vs. voting?

4. **Regional Expansion**:
   - How to adapt to other countries?
   - Shared vs. per-country models?

5. **Explainability**:
   - SHAP values for model explanations?
   - Feature attribution for policy makers?

---

## Decision Log Template

For future decisions, use this template:

```markdown
### Decision: [Title]

**Context**: [Why decision was needed]

**Decision**: [What was chosen]

**Reasoning**: [Why this over alternatives]

**Implementation**: [Technical details]

**Trade-offs**:
- ✓ Pros
- ✗ Cons

**Related**: [Links to related decisions]

**Review Date**: [When to revisit]
```

---

## References

- [Architecture Document](./ARCHITECTURE.md) - System design
- [Project Handbook](./PROJECT_HANDBOOK.md) - Development guide
- [API Documentation](./API.md) - Endpoint reference
- [README](../README.md) - Project overview

---

**Last Updated:** February 2026  
**Decisions Made By:** Development Team  
**Next Review:** Q2 2026
