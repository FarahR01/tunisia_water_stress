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

- [ ] Code follows project style
- [ ] Tests pass locally
- [ ] No debug print statements
- [ ] Meaningful commit message
- [ ] Changes are focused and small
- [ ] No credentials or sensitive data

---

For questions or issues, refer to the main README.md
