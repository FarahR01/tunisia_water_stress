import argparse
import os
import random
from pprint import pprint

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
    XGBRegressor = None

try:
    from .config_train import TrainConfig
    from .data_loader import list_available_indicators, load_and_pivot
    from .data_versioning import DataManifest
    from .evaluate import (
        plot_actual_vs_pred,
        plot_feature_importance,
        regression_metrics,
        save_metrics,
    )
    from .feature_engineering import add_lag_features, add_year_column
    from .feature_importance import (
        extract_feature_importance,
        plot_top_features,
        save_feature_importance_summary,
    )
    from .hyperparameter_tuning import save_hyperparameter_results, tune_hyperparameters
    from .mlflow_utils import MLflowTracker
    from .preprocessing import drop_sparse_columns, fill_missing
except Exception:
    # Support running `python src/train.py` (script) where `src/` is on sys.path
    from config_train import TrainConfig
    from data_loader import list_available_indicators, load_and_pivot
    from data_versioning import DataManifest
    from evaluate import (
        plot_actual_vs_pred,
        plot_feature_importance,
        regression_metrics,
        save_metrics,
    )
    from feature_engineering import add_lag_features, add_year_column
    from feature_importance import (
        extract_feature_importance,
        plot_top_features,
        save_feature_importance_summary,
    )
    from hyperparameter_tuning import save_hyperparameter_results, tune_hyperparameters
    from mlflow_utils import MLflowTracker
    from preprocessing import drop_sparse_columns, fill_missing


print(">>> USING UPDATED TRAIN FILE (leakage fix applied) <<<")


# ============================================================================
# REPRODUCIBILITY: SET RANDOM SEEDS
# ============================================================================
def set_random_seeds(config: "TrainConfig") -> None:
    """Set all random seeds for reproducibility.

    Args:
        config: TrainConfig with seed settings
    """
    np.random.seed(config.seeds.numpy_seed)
    random.seed(config.seeds.python_seed)
    print(f"✓ Random seeds set: numpy={config.seeds.numpy_seed}, python={config.seeds.python_seed}")


# ============================================================================
# DATA VERSIONING: TRACK DATA FILES
# ============================================================================
def track_data_versions(config: "TrainConfig") -> None:
    """Track data file versions for reproducibility.

    Args:
        config: TrainConfig with data paths
    """
    try:
        manifest = DataManifest()
        if os.path.exists(config.data.raw_data_path):
            manifest.add_file(config.data.raw_data_path, source="World Bank")
        if os.path.exists(config.data.processed_data_path):
            manifest.add_file(config.data.processed_data_path, source="Pipeline")

        manifest_path = os.path.join(
            os.path.dirname(config.data.raw_data_path), "DATA_MANIFEST.json"
        )
        manifest.save(manifest_path)
        print(f"✓ Data manifest saved to {manifest_path}")
    except Exception as e:
        print(f"Warning: Could not create data manifest: {e}")


DEFAULT_FEATURE_KEYWORDS = [
    "renewable",
    "water productivity",
    "population",
    "urban",
    "rural",
    "agricultural land",
    "arable",
    "precipitation",
    "forest",
    "greenhouse",
]

# Blacklist features that cause leakage or mathematical dependence with water-stress target
FEATURE_BLACKLIST = [
    "annual freshwater withdrawals, agriculture",
    "annual freshwater withdrawals, domestic",
    "annual freshwater withdrawals, industry",
    "annual freshwater withdrawals, total",
    "level of water stress",  # this is the target itself
]


def choose_columns_by_keywords(columns, keywords, max_features=12):
    cols = []
    cols_lower = [c.lower() for c in columns]
    for kw in keywords:
        for i, c in enumerate(columns):
            # skip blacklisted features
            if any(bl in cols_lower[i] for bl in FEATURE_BLACKLIST):
                continue
            if kw in cols_lower[i] and c not in cols:
                cols.append(c)
                if len(cols) >= max_features:
                    return cols
    return cols


def choose_target(columns):
    # Prioritize "Level of water stress" as the primary target
    cols_lower = [c.lower() for c in columns]
    # First pass: look for exact "water stress" indicator
    for i, c in enumerate(columns):
        if "water stress" in cols_lower[i]:
            return c
    # Second pass: any freshwater/withdraw indicator (fallback)
    for i, c in enumerate(columns):
        name = cols_lower[i]
        if "freshwater" in name or "withdraw" in name:
            return c
    return None


def temporal_split(df: pd.DataFrame, train_end: int = 2010):
    train = df[df.index <= train_end]
    test = df[df.index > train_end]
    return train, test


def train_and_evaluate(
    X_train,
    y_train,
    X_test,
    y_test,
    out_dir: str,
    config: "TrainConfig" = None,
    mlflow_logger=None,
    tune_hyperparams=False,
    use_ridge=False,
    use_lasso=False,
    use_xgboost=False,
):
    """Train and evaluate models with optional MLflow tracking.

    Args:
        X_train, y_train: Training features and target
        X_test, y_test: Test features and target
        out_dir: Directory to save model artifacts
        config: TrainConfig for reproducibility logging
        mlflow_logger: MLflow logger for experiment tracking
        tune_hyperparams: Enable hyperparameter tuning
        use_ridge, use_lasso, use_xgboost: Model selection flags
    """
    # Use config settings if provided, otherwise fall back to flags
    if config:
        tune_hyperparams = config.models.enable_hyperparameter_tuning
        use_ridge = config.models.ridge.enabled
        use_lasso = config.models.lasso.enabled
        use_xgboost = config.models.xgboost.enabled

    # Scale features for linear models to improve numeric stability (Ridge/Lasso)
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    try:
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), index=X_test.index, columns=X_test.columns
        )
    except Exception:
        # fallback: if conversion fails, continue with originals
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(
            random_state=42 if not config else config.seeds.sklearn_seed
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, random_state=42 if not config else config.seeds.sklearn_seed
        ),
    }

    # Add Ridge/Lasso if requested
    if use_ridge:
        models["Ridge"] = Ridge()
    if use_lasso:
        # increase max_iter to help convergence on Lasso
        models["Lasso"] = Lasso(max_iter=10000)
    if use_xgboost:
        if XGBOOST_AVAILABLE and XGBRegressor is not None:
            try:
                seed = 42 if not config else config.seeds.xgboost_seed
                models["XGBoost"] = XGBRegressor(random_state=seed, verbosity=0)
                print("✓ XGBoost successfully added to models")
            except Exception as e:
                print(f"Warning: Failed to instantiate XGBoost: {e}")
        else:
            print("Warning: XGBoost requested but not available. Install with: pip install xgboost")

    results = {}
    tuning_results = {}
    importance_dfs = []

    os.makedirs(out_dir, exist_ok=True)

    for name, model in models.items():
        # choose scaled inputs for linear models
        if name in ("LinearRegression", "Ridge", "Lasso"):
            Xtr = X_train_scaled
            Xte = X_test_scaled
        else:
            Xtr = X_train
            Xte = X_test

        # Hyperparameter tuning if enabled
        if tune_hyperparams:
            tuning_result = tune_hyperparameters(Xtr, y_train, Xte, y_test, name, model)
            tuning_results[name] = tuning_result
            model = tuning_result["best_model"]
        else:
            model.fit(Xtr, y_train)

        y_pred = model.predict(Xte)
        metrics = regression_metrics(y_test, y_pred)
        results[name] = metrics
        joblib.dump(model, os.path.join(out_dir, f"{name}.joblib"))

        # Log to MLflow if logger provided
        if mlflow_logger:
            for metric_name, metric_value in metrics.items():
                mlflow_logger.log_metric(f"{name}/{metric_name}", metric_value)

        # plots for predictions
        plot_path = os.path.join(out_dir, f"{name}_actual_vs_pred.png")
        plot_actual_vs_pred(X_test.index, y_test, y_pred, plot_path)
        if mlflow_logger:
            mlflow_logger.log_artifact(plot_path, "artifacts/predictions")

        try:
            imp_path = os.path.join(out_dir, f"{name}_feature_importance.png")
            plot_feature_importance(
                model,
                X_train.columns.tolist(),
                imp_path,
            )
            if mlflow_logger:
                mlflow_logger.log_artifact(imp_path, "artifacts/feature_importance")
        except Exception:
            pass

        # Extract feature importance
        importance_df = extract_feature_importance(model, X_train.columns.tolist(), name)
        if importance_df is not None:
            importance_dfs.append(importance_df)

    save_metrics(results, os.path.join(out_dir, "metrics.csv"))
    if mlflow_logger:
        mlflow_logger.log_artifact(os.path.join(out_dir, "metrics.csv"), "artifacts")

    # Save feature importance summary
    if importance_dfs:
        importance_path = os.path.join(out_dir, "feature_importance_summary.csv")
        save_feature_importance_summary(importance_dfs, importance_path)
        if mlflow_logger:
            mlflow_logger.log_artifact(importance_path, "artifacts")

        # Create comparison plot
        plot_top_features(models, X_train.columns.tolist(), X_test, top_n=10, out_dir=out_dir)

    # Save hyperparameter tuning results
    if tuning_results:
        save_hyperparameter_results(tuning_results, out_dir)
        if mlflow_logger:
            mlflow_logger.log_artifact(
                os.path.join(out_dir, "hyperparameter_tuning_summary.csv"), "artifacts"
            )

    return results


def main(args):
    """Main training orchestration function.

    Args:
        args: Argparse namespace or TrainConfig instance
    """
    # Support both CLI flags (legacy) and config file (new)
    config = None
    if hasattr(args, "config") and args.config:
        # Config file provided
        config = TrainConfig.from_yaml(args.config)
        print(f"✓ Loaded config from {args.config}")
    else:
        # Legacy CLI mode: construct config from args
        config = args_to_config(args)

    # Set reproducibility seeds first
    set_random_seeds(config)

    # Track data versions
    track_data_versions(config)

    # Initialize MLflow tracker
    tracker = MLflowTracker.from_config(config.mlflow) if config.mlflow.enabled else None
    mlflow_logger = None

    if tracker:
        tracker.start_run(
            run_name=f"train_{config.output_dir.split('/')[-1]}",
            tags={"framework": "scikit-learn", "config_used": str(config)},
        )
        mlflow_logger = tracker.get_logger()

        # Log config parameters
        config_dict = config.model_dump(by_alias=False)
        mlflow_logger.log_config(config_dict)

        # Save config to MLflow artifacts
        config_artifact_path = os.path.join(config.output_dir, "config_used.yaml")
        config.to_yaml(config_artifact_path)

    try:
        raw = config.data.raw_data_path
        processed = config.data.processed_data_path
        os.makedirs(os.path.dirname(processed), exist_ok=True)
        if not os.path.exists(processed):
            print("Processing raw CSV to wide form...")
            df = load_and_pivot(raw, processed)
        else:
            df = pd.read_csv(processed, index_col=0)
            df.index = df.index.astype(int)

        print(f"Data has {df.shape[0]} years and {df.shape[1]} indicators")

        # list available indicators
        inds = list_available_indicators(raw)
        print("Available indicators (sample):")
        pprint(inds[:30])

        # choose target
        target = config.data.target_column
        if not target:
            target = choose_target(df.columns.tolist())
        if not target:
            raise SystemExit(
                "Could not auto-detect a target indicator. Please pass target_column in config."
            )
        print(f"Using target indicator: {target}")

        # choose features
        if config.data.explicit_features:
            features = config.data.explicit_features
        else:
            features = choose_columns_by_keywords(
                df.columns.tolist(),
                config.data.feature_keywords,
                max_features=config.feature_engineering.max_features,
            )

        # CRITICAL: Remove target from features to prevent data leakage
        features = [f for f in features if f != target]

        print("Selected features:")
        pprint(features)

        # prepare dataset
        data = df.copy()
        # keep target and features
        cols_needed = [c for c in features if c in data.columns]
        if target not in data.columns:
            raise SystemExit(f"Target '{target}' not present in processed data columns")
        cols_needed = cols_needed + [target]
        data = data.loc[:, cols_needed]

        data = drop_sparse_columns(data, threshold=config.data.sparse_threshold)
        data = fill_missing(data)

        # Optional: automatic leakage filtering (drop features highly correlated with target)
        if config.feature_engineering.apply_leakage_filter:
            try:
                corr = data.corr().abs()
                if target in corr.columns:
                    target_corr = corr[target].drop(labels=[target], errors="ignore")
                    leaking = target_corr[
                        target_corr >= config.feature_engineering.leakage_correlation_threshold
                    ].index.tolist()
                    if leaking:
                        threshold = config.feature_engineering.leakage_correlation_threshold
                        print(
                            f"Dropping {len(leaking)} features due to high correlation "
                            f"with target (>= {threshold}):"
                        )
                        for c in leaking:
                            print("  -", c)
                        data = data.drop(columns=leaking)
            except Exception as e:
                print(f"Warning: leakage filtering skipped due to error: {e}")

        # Optional: pairwise collinearity filtering (drop one of each highly-correlated pair)
        if config.feature_engineering.apply_collinearity_filter:
            try:
                # consider only feature columns (exclude target)
                feature_cols = [c for c in data.columns if c != target]
                if feature_cols:
                    corr_mat = data[feature_cols].corr().abs()
                    # create list of pairs (i, j, corr) for i<j
                    pairs = []
                    for i_idx, i in enumerate(corr_mat.index):
                        for j_idx, j in enumerate(corr_mat.columns):
                            if j_idx <= i_idx:
                                continue
                            pairs.append((i, j, float(corr_mat.iat[i_idx, j_idx])))
                    # sort by correlation descending so we drop the more redundant columns first
                    pairs.sort(key=lambda x: x[2], reverse=True)
                    to_drop = set()
                    thresh = config.feature_engineering.collinearity_threshold
                    for i, j, val in pairs:
                        if val >= thresh:
                            # drop j if neither already marked
                            if i not in to_drop and j not in to_drop:
                                to_drop.add(j)
                    if to_drop:
                        thresh = config.feature_engineering.collinearity_threshold
                        print(
                            f"Dropping {len(to_drop)} features due to high pairwise "
                            f"collinearity (>= {thresh}):"
                        )
                        for c in sorted(to_drop):
                            print("  -", c)
                        data = data.drop(columns=list(to_drop))
                        # save dropped list to models dir for auditing
                        try:
                            os.makedirs(config.output_dir, exist_ok=True)
                            with open(
                                os.path.join(config.output_dir, "collinearity_dropped.txt"),
                                "w",
                                encoding="utf-8",
                            ) as fh:
                                for c in sorted(to_drop):
                                    fh.write(c + "\n")
                        except Exception:
                            pass
            except Exception as e:
                print(f"Warning: collinearity filtering skipped due to error: {e}")

        # optional lag features
        if config.feature_engineering.lag_features > 0:
            data = add_lag_features(
                data,
                [c for c in features if c in data.columns],
                lags=config.feature_engineering.lag_features,
            )

        data = add_year_column(data)
        data.set_index(data.index, inplace=True)

        # final split
        y = data[target]
        # ensure y is a DataFrame (handle duplicate-column or Series cases)
        if isinstance(y, pd.Series):
            y = y.to_frame(name=target)
        X = data.drop(columns=[target])

        X_train_df, X_test_df = temporal_split(X, train_end=config.split.train_end_year)
        y_train, y_test = temporal_split(y, train_end=config.split.train_end_year)
        y_train = y_train[target]
        y_test = y_test[target]

        print(
            f"Train years: {X_train_df.index.min()}-{X_train_df.index.max()} | "
            f"Test years: {X_test_df.index.min()}-{X_test_df.index.max()}"
        )

        results = train_and_evaluate(
            X_train_df,
            y_train,
            X_test_df,
            y_test,
            out_dir=config.output_dir,
            config=config,
            mlflow_logger=mlflow_logger,
        )
        print("Training complete. Metrics:")
        pprint(results)

        # Log final results
        if mlflow_logger:
            mlflow_logger.log_metrics({"training_complete": 1})

    finally:
        if tracker:
            tracker.end_run()


def args_to_config(args) -> "TrainConfig":
    """Convert argparse args to TrainConfig for backward compatibility.

    Args:
        args: Argparse namespace with old CLI flags

    Returns:
        TrainConfig instance
    """
    from config_train import (
        DataConfig,
        FeatureEngineeringConfig,
        MLflowConfig,
        ModelConfig,
        ModelsConfig,
        SeedConfig,
        SplitConfig,
        TrainConfig,
    )

    # Extract feature list if provided
    explicit_features = None
    if hasattr(args, "features") and args.features:
        explicit_features = [f.strip() for f in args.features.split(",")]

    config = TrainConfig(
        seeds=SeedConfig(numpy_seed=42, python_seed=42, sklearn_seed=42, xgboost_seed=42),
        data=DataConfig(
            raw_data_path=args.raw,
            processed_data_path=args.processed,
            explicit_features=explicit_features,
            target_column=getattr(args, "target", None),
        ),
        feature_engineering=FeatureEngineeringConfig(
            lag_features=int(args.lags),
            apply_leakage_filter=getattr(args, "leakage_filter", False),
            leakage_correlation_threshold=float(getattr(args, "leakage_threshold", 0.99)),
            apply_collinearity_filter=getattr(args, "collinearity_filter", False),
            collinearity_threshold=float(getattr(args, "collinearity_threshold", 0.95)),
        ),
        split=SplitConfig(train_end_year=args.train_end),
        models=ModelsConfig(
            enable_hyperparameter_tuning=getattr(args, "tune_hyperparams", False),
            ridge=ModelConfig(enabled=getattr(args, "use_ridge", False)),
            lasso=ModelConfig(enabled=getattr(args, "use_lasso", False)),
            xgboost=ModelConfig(enabled=getattr(args, "use_xgboost", False)),
        ),
        mlflow=MLflowConfig(enabled=False),  # Disable for old CLI mode
        output_dir=args.models_dir,
    )

    print("ℹ Using legacy CLI mode. Consider using --config for reproducibility.")
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train water-stress models for Tunisia.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using new config-based system (RECOMMENDED)
  python src/train.py --config config/train_config.yaml

  # Legacy CLI mode (for backward compatibility)
  python src/train.py --processed data/processed/processed_tunisia.csv \\
      --models_dir models/ --tune_hyperparams
        """,
    )

    # New config-based interface
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (recommended for reproducibility)",
    )

    # Legacy CLI arguments (kept for backward compatibility)
    parser.add_argument("--raw", default=os.path.join("data", "raw", "environment_tun.csv"))
    parser.add_argument(
        "--processed", default=os.path.join("data", "processed", "processed_tunisia.csv")
    )
    parser.add_argument("--models_dir", default="models/")
    parser.add_argument("--target", default=None, help="Exact indicator name to use as target")
    parser.add_argument(
        "--features", default=None, help="Comma-separated list of feature indicator names to use"
    )
    parser.add_argument("--lags", default=0, help="Number of lag features to add (default 0)")
    parser.add_argument(
        "--train_end", type=int, default=2010, help="Last year to include in training split"
    )
    parser.add_argument(
        "--leakage_filter",
        action="store_true",
        help="Drop features with absolute correlation >= leakage_threshold",
    )
    parser.add_argument(
        "--leakage_threshold",
        type=float,
        default=0.99,
        help="Correlation threshold for leakage filtering (default 0.99)",
    )
    parser.add_argument(
        "--collinearity_filter",
        action="store_true",
        help="Drop features with high pairwise correlation >= collinearity_threshold",
    )
    parser.add_argument(
        "--collinearity_threshold",
        type=float,
        default=0.95,
        help="Pairwise correlation threshold for collinearity filtering (default 0.95)",
    )
    parser.add_argument(
        "--tune_hyperparams",
        action="store_true",
        help="Enable hyperparameter tuning using GridSearchCV",
    )
    parser.add_argument(
        "--use_ridge", action="store_true", help="Include Ridge regression in models"
    )
    parser.add_argument(
        "--use_lasso", action="store_true", help="Include Lasso regression in models"
    )
    parser.add_argument(
        "--use_xgboost",
        action="store_true",
        help="Include XGBoost in models (requires xgboost package)",
    )
    args = parser.parse_args()
    main(args)
