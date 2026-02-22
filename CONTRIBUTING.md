# Contributing Guide

## Git Workflow

Follow these practices for professional, maintainable code.

### Commit Message Format

```
type(scope): short description

Optional detailed explanation if needed.
```

**Example commits:**
- `feat(preprocessing): add outlier detection for water stress data`
- `fix(models): correct feature scaling in LinearRegression`
- `refactor(train): optimize hyperparameter tuning pipeline`
- `docs(README): update installation instructions`
- `chore(deps): update scikit-learn to 1.2.0`
- `experiment(models): test XGBoost on Tunisia dataset`
- `data(raw): add Q1 2025 monitoring records`

### Commit Types

| Type | When to use | Example |
|------|-------------|---------|
| **feat** | New feature | New model, new preprocessing step |
| **fix** | Bug fix | Correct data leakage, fix calculation |
| **refactor** | Code restructuring | Reorganize functions, improve performance |
| **docs** | Documentation | README, docstrings, comments |
| **chore** | Minor maintenance | Dependency updates, formatting |
| **experiment** | Model testing | Test new model architecture |
| **data** | Dataset changes | Add/update raw data, clean datasets |

### Branch Naming Convention

```
type/short-description
```

**Examples:**
- `feat/water-stress-prediction-model`
- `fix/feature-scaling-bug`
- `refactor/pipeline-optimization`
- `chore/update-dependencies`
- `experiment/ensemble-models`

### Branch Strategy

1. **Main branch**: Production-ready code with stable models
2. **Develop branch**: Integration branch for features
3. **Feature/fix branches**: Individual work

```
main
 ├── develop
 │   ├── feat/preprocessing-improvements
 │   ├── fix/data-validation
 │   └── experiment/model-comparison
```

### Workflow Steps

#### 1. Create a Feature Branch
```bash
git checkout develop
git pull origin develop
git checkout -b feat/your-feature-name
```

#### 2. Make Logical Commits
- One feature/fix per commit
- Keep commits small and focused
- Test before committing

```bash
git add .
git commit -m "feat(scope): clear description"
```

#### 3. Push Regularly
```bash
git push origin feat/your-feature-name
```

#### 4. Create Pull Request
- Add description explaining changes
- Link related issues
- Request review

#### 5. Merge to Develop
- Ensure all tests pass
- Squash if multiple experimental commits
- Delete feature branch after merge

```bash
git switch develop
git pull origin develop
git merge --no-ff feat/your-feature-name
git push origin develop
git branch -d feat/your-feature-name
```

### Golden Rules ✔

1. **One logical change per commit** - Each commit should represent one complete idea
2. **Commit messages in English** - Always use English for consistency
3. **Never commit `venv/`** - Already in .gitignore
4. **Never commit raw temporary files** - Use .gitignore for data, models, cache
5. **Push often** - Daily commits prevent large merge conflicts
6. **Test before committing** - Run validation scripts before staging
7. **Write descriptive messages** - Future you will thank you
8. **Keep commits atomic** - Can revert single commits without breaking code

### Files to Never Commit

```
❌ venv/                           (Virtual environment)
❌ __pycache__/                    (Python cache)
❌ *.joblib, *.pkl                 (Trained models)
❌ data/raw/, data/processed/      (Raw/processed data)
❌ .ipynb_checkpoints              (Jupyter cache)
❌ *.log, *.tmp                    (Temporary files)
```

### Useful Git Commands

```bash
# View untracked files that should be ignored
git status

# View commit history
git log --oneline -10

# Amend last commit (before pushing!)
git commit --amend --no-edit

# Undo local changes
git restore <file>

# View changes before staging
git diff <file>

# Squash multiple commits
git rebase -i HEAD~3
```

### Pre-commit Checklist

- [ ] Code follows project style (black formatting)
- [ ] Tests pass locally (100% coverage on src/)
- [ ] Type hints added to critical functions
- [ ] No debug print statements
- [ ] Meaningful, detailed commit message
- [ ] Changes are focused and small
- [ ] No credentials or sensitive data
- [ ] Docstrings include purpose, args, returns
- [ ] Flake8 lint passes (0 errors)
- [ ] MyPy type checking passes

---

## Code Quality Requirements

### 1. Type Hints (PEP 484)

All functions in `src/` and `api/` must include type hints:

**Good:**
```python
from typing import Optional, List, Dict
import pandas as pd

def drop_sparse_columns(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Drop columns with more than threshold% missing values.
    
    Args:
        df: Input DataFrame
        threshold: Missing value fraction threshold (0-1)
    
    Returns:
        DataFrame with sparse columns removed
    """
    missing_pct = df.isnull().sum() / len(df)
    cols_to_keep = missing_pct[missing_pct <= threshold].index
    return df[cols_to_keep]
```

**Required format:**
- All parameters: `param: Type`
- Return value: `-> ReturnType`
- Complex types: Use `typing` module (List, Dict, Optional, etc.)
- Docstrings with Args, Returns sections

### 2. Black Code Formatting

All code must pass black formatting check:

```bash
# Format your changes
python -m black src/ api/ tests/ --line-length=100

# Check before committing
python -m black --check src/ api/ tests/ --line-length=100
```

Rules:
- Line length: 100 characters max
- String quotes: Double quotes preferred
- Trailing commas: Added in multi-line structures

### 3. Flake8 Linting

Zero flake8 errors required in `src/` and `api/`:

```bash
python -m flake8 src/ api/ --max-line-length=100
```

Common fixes:
- Remove unused imports: `import X  # noqa: F401` or delete
- Unused variables: Use `_` (underscore)
- Line too long: Break into multiple lines
- Blank lines: Max 2 consecutive

### 4. MyPy Type Checking

Static type checking must pass:

```bash
python -m mypy src/ --ignore-missing-imports
```

Common fixes:
- Optional types: `Optional[str]` instead of just `str`
- Union types: `Union[int, float]` for multiple types
- Callable types: `Callable[[int], str]` for functions
- TypedDict for complex dictionaries

### 5. Docstrings

Every function must have a docstring (Google style):

```python
def load_and_pivot(
    raw_csv_path: str, 
    processed_path: Optional[str] = None
) -> pd.DataFrame:
    """Load World Bank CSV data and pivot to wide format.
    
    Converts long-format World Bank data (years, indicators, values)
    to wide format (years as index, indicators as columns).
    
    Args:
        raw_csv_path: Path to World Bank CSV file
        processed_path: Optional path to save processed data
    
    Returns:
        DataFrame with years as index, indicators as columns
    
    Raises:
        FileNotFoundError: If raw_csv_path does not exist
        ValueError: If CSV format is invalid
    
    Examples:
        >>> df = load_and_pivot('data/raw/env_tun.csv')
        >>> df.shape
        (65, 47)
    """
    # Implementation here
    pass
```

### 6. Testing Requirements

**Minimum coverage: 100% on critical modules** (`src/` only)

```bash
pytest tests/ --cov=src/ --cov-report=term-missing
```

Each new function needs a test:

```python
def test_drop_sparse_columns():
    """Test that sparse columns are dropped correctly."""
    df = pd.DataFrame({
        'full': [1, 2, 3],
        'sparse': [1, None, None]  # 67% missing
    })
    result = drop_sparse_columns(df, threshold=0.5)
    assert 'sparse' not in result.columns
    assert 'full' in result.columns
```

**Test Structure:**
1. Arrange: Set up test data
2. Act: Call function
3. Assert: Verify result

### 7. Pre-commit Hooks

Hooks run automatically before each commit:

```bash
# Install hooks
pre-commit install

# Run manually (before committing)
pre-commit run --all-files
```

**Included hooks:**
- `black` - Code formatting
- `flake8` - Linting
- `mypy` - Type checking
- `isort` - Import sorting
- `bandit` - Security scanning
- `trailing-whitespace` - Remove trailing spaces
- `end-of-file-fixer` - Ensure newline at EOF
- `check-yaml` - Validate YAML syntax

If a hook fails:
1. Fix the error (usually automatic)
2. Stage the fixed files
3. Commit again

---

## Testing Guide

### Running Tests

```bash
# All tests with coverage
python -m pytest tests/ -v --cov=src/ --cov-report=html

# Specific test file
python -m pytest tests/test_preprocessing.py -v

# Specific test class
python -m pytest tests/test_preprocessing.py::TestFillMissing -v

# Specific test function
python -m pytest tests/test_preprocessing.py::TestFillMissing::test_fill_forward -v

# With debugging (stop on failure)
python -m pytest tests/ -v --pdb

# Coverage report (HTML)
python -m pytest tests/ --cov=src/ --cov-report=html
open htmlcov/index.html  # View in browser
```

### Test Organization

```
tests/
├── test_data_loader.py          # Tests for data_loader.py
│   ├── TestLoadAndPivot         # Test class 1
│   │   ├── test_...             # Test method 1
│   │   └── test_...             # Test method 2
│   └── TestListAvailableIndicators  # Test class 2
│
├── test_preprocessing.py        # Tests for preprocessing.py
│   ├── TestDropSparseColumns
│   ├── TestFillMissing
│   └── TestSelectFeatures
│
├── test_feature_engineering.py  # Tests for feature_engineering.py
│   ├── TestAddLagFeatures
│   └── TestAddYearColumn
│
├── test_pipeline_integration.py # End-to-end pipeline tests
│   ├── TestPipelineDataLoad
│   ├── TestPipelinePreprocessing
│   └── TestPipelineTrainEvaluate
│
└── conftest.py                  # Shared fixtures
```

### Writing Tests

**Template:**
```python
import pytest
import pandas as pd
from src.preprocessing import drop_sparse_columns

class TestDropSparseColumns:
    """Tests for drop_sparse_columns function."""
    
    def test_drops_sparse_columns_above_threshold(self):
        """Test that columns >50% missing are dropped."""
        df = pd.DataFrame({
            'full': [1, 2, 3],
            'mostly_empty': [1, None, None]
        })
        result = drop_sparse_columns(df, threshold=0.50)
        assert 'mostly_empty' not in result.columns
        assert len(result.columns) == 1
    
    def test_keeps_dense_columns(self):
        """Test that columns <threshold% missing are kept."""
        df = pd.DataFrame({
            'dense': [1, 2, 3, 4, 5],
            'one_missing': [1, 2, 3, None, 5]
        })
        result = drop_sparse_columns(df, threshold=0.5)
        assert 'dense' in result.columns
        assert 'one_missing' in result.columns
    
    def test_handles_empty_dataframe(self):
        """Test behavior with empty DataFrame."""
        df = pd.DataFrame()
        result = drop_sparse_columns(df)
        assert result.empty
```

---

## Development Workflow

### Step-by-Step: Adding a Feature

1. **Create feature branch:**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feat/my-feature
   ```

2. **Implement feature with tests:**
   ```python
   # src/my_module.py
   def my_function(x: int) -> str:
       """My new feature."""
       return str(x)
   
   # tests/test_my_module.py
   def test_my_function():
       assert my_function(5) == "5"
   ```

3. **Run quality checks:**
   ```bash
   # Format
   python -m black src/ tests/
   
   # Lint
   python -m flake8 src/ tests/
   
   # Type check
   python -m mypy src/
   
   # Test
   python -m pytest tests/ -v --cov=src/
   ```

4. **Commit with clear message:**
   ```bash
   git add src/my_module.py tests/test_my_module.py
   git commit -m "feat(my_module): add my feature
   
   - Implement my_function
   - Add comprehensive tests
   - Update docstrings"
   ```

5. **Push and create PR:**
   ```bash
   git push origin feat/my-feature
   # Create PR on GitHub with description
   ```

6. **Code review & merge:**
   - Request review from maintainers
   - Address feedback
   - Merge to develop once approved
   - Delete feature branch

---

## Useful Git Commands

```bash
# View untracked files that should be ignored
git status

# View commit history
git log --oneline -10

# Amend last commit (before pushing!)
git commit --amend --no-edit

# Undo local changes
git restore <file>

# View changes before staging
git diff <file>

# Squash multiple commits
git rebase -i HEAD~3

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Push force (use with caution!)
git push origin feat/my-feature --force-with-lease
```

### Pre-commit Checklist

- [ ] Code follows project style (black formatting)
- [ ] Tests pass locally (100% coverage on src/)
- [ ] Type hints added to critical functions
- [ ] No debug print statements
- [ ] Meaningful, detailed commit message
- [ ] Changes are focused and small
- [ ] No credentials or sensitive data
- [ ] Docstrings include purpose, args, returns
- [ ] Flake8 lint passes (0 errors)
- [ ] MyPy type checking passes

---

## Getting Help

- **Questions?** Check [PROJECT_HANDBOOK.md](docs/PROJECT_HANDBOOK.md)
- **Design decisions?** See [DECISIONS.md](docs/DECISIONS.md)
- **System architecture?** Review [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Issues?** Open GitHub issue with details
- **Code review?** Request from maintainers in PR

---

**Last Updated:** February 2026  
**Maintained by:** Development Team
