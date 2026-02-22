import argparse
import os
from pprint import pprint

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBOOST_AVAILABLE = False
    XGBRegressor = None

try:
    from .data_loader import load_and_pivot, list_available_indicators
    from .evaluate import (
        plot_actual_vs_pred,
        plot_feature_importance,
        regression_metrics,
        save_metrics,
    )
    from .feature_engineering import add_lag_features, add_year_column
    from .preprocessing import drop_sparse_columns, fill_missing, select_features
    from .feature_importance import (
        extract_feature_importance,
        plot_top_features,
        save_feature_importance_summary,
    )
    from .hyperparameter_tuning import tune_hyperparameters, save_hyperparameter_results
except Exception:
    # Support running `python src/train.py` (script) where `src/` is on sys.path
    from data_loader import load_and_pivot, list_available_indicators
    from evaluate import (
        plot_actual_vs_pred,
        plot_feature_importance,
        regression_metrics,
        save_metrics,
    )
    from feature_engineering import add_lag_features, add_year_column
    from preprocessing import drop_sparse_columns, fill_missing, select_features
    from feature_importance import (
        extract_feature_importance,
        plot_top_features,
        save_feature_importance_summary,
    )
    from hyperparameter_tuning import tune_hyperparameters, save_hyperparameter_results


print(">>> USING UPDATED TRAIN FILE (leakage fix applied) <<<")

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
    tune_hyperparams=False,
    use_ridge=False,
    use_lasso=False,
    use_xgboost=False,
):
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
        "DecisionTree": DecisionTreeRegressor(random_state=0),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=0),
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
                models["XGBoost"] = XGBRegressor(random_state=0, verbosity=0)
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

        # plots for predictions
        plot_actual_vs_pred(
            X_test.index, y_test, y_pred, os.path.join(out_dir, f"{name}_actual_vs_pred.png")
        )
        try:
            plot_feature_importance(
                model,
                X_train.columns.tolist(),
                os.path.join(out_dir, f"{name}_feature_importance.png"),
            )
        except Exception:
            pass

        # Extract feature importance
        importance_df = extract_feature_importance(model, X_train.columns.tolist(), name)
        if importance_df is not None:
            importance_dfs.append(importance_df)

    save_metrics(results, os.path.join(out_dir, "metrics.csv"))

    # Save feature importance summary
    if importance_dfs:
        save_feature_importance_summary(
            importance_dfs, os.path.join(out_dir, "feature_importance_summary.csv")
        )
        # Create comparison plot
        plot_top_features(models, X_train.columns.tolist(), X_test, top_n=10, out_dir=out_dir)

    # Save hyperparameter tuning results
    if tuning_results:
        save_hyperparameter_results(tuning_results, out_dir)

    return results


def main(args):
    raw = args.raw
    processed = args.processed
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
    target = args.target
    if not target:
        target = choose_target(df.columns.tolist())
    if not target:
        raise SystemExit(
            "Could not auto-detect a target indicator. Please pass --target with the exact indicator name."
        )
    print(f"Using target indicator: {target}")

    # choose features
    if args.features:
        features = [f.strip() for f in args.features.split(",")]
    else:
        features = choose_columns_by_keywords(
            df.columns.tolist(), DEFAULT_FEATURE_KEYWORDS, max_features=12
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

    data = drop_sparse_columns(data, threshold=0.5)
    data = fill_missing(data)

    # Optional: automatic leakage filtering (drop features highly correlated with target)
    if getattr(args, "leakage_filter", False):
        try:
            corr = data.corr().abs()
            if target in corr.columns:
                target_corr = corr[target].drop(labels=[target], errors="ignore")
                leaking = target_corr[target_corr >= float(args.leakage_threshold)].index.tolist()
                if leaking:
                    print(
                        f"Dropping {len(leaking)} features due to high correlation with target (>= {args.leakage_threshold}):"
                    )
                    for c in leaking:
                        print("  -", c)
                    data = data.drop(columns=leaking)
        except Exception as e:
            print(f"Warning: leakage filtering skipped due to error: {e}")

    # Optional: pairwise collinearity filtering (drop one of each highly-correlated pair)
    if getattr(args, "collinearity_filter", False):
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
                thresh = float(args.collinearity_threshold)
                for i, j, val in pairs:
                    if val >= thresh:
                        # drop j if neither already marked
                        if i not in to_drop and j not in to_drop:
                            to_drop.add(j)
                if to_drop:
                    print(
                        f"Dropping {len(to_drop)} features due to high pairwise collinearity (>= {thresh}):"
                    )
                    for c in sorted(to_drop):
                        print("  -", c)
                    data = data.drop(columns=list(to_drop))
                    # save dropped list to models dir for auditing
                    try:
                        os.makedirs(args.models_dir, exist_ok=True)
                        with open(
                            os.path.join(args.models_dir, "collinearity_dropped.txt"),
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
    if args.lags and int(args.lags) > 0:
        data = add_lag_features(
            data, [c for c in features if c in data.columns], lags=int(args.lags)
        )

    data = add_year_column(data)
    data.set_index(data.index, inplace=True)

    # final split
    y = data[target]
    # ensure y is a DataFrame (handle duplicate-column or Series cases)
    if isinstance(y, pd.Series):
        y = y.to_frame(name=target)
    X = data.drop(columns=[target])

    X_train_df, X_test_df = temporal_split(X, train_end=args.train_end)
    y_train, y_test = temporal_split(y, train_end=args.train_end)
    y_train = y_train[target]
    y_test = y_test[target]

    print(
        f"Train years: {X_train_df.index.min()}-{X_train_df.index.max()} | Test years: {X_test_df.index.min()}-{X_test_df.index.max()}"
    )

    results = train_and_evaluate(
        X_train_df,
        y_train,
        X_test_df,
        y_test,
        out_dir=args.models_dir,
        tune_hyperparams=getattr(args, "tune_hyperparams", False),
        use_ridge=getattr(args, "use_ridge", False),
        use_lasso=getattr(args, "use_lasso", False),
        use_xgboost=getattr(args, "use_xgboost", False),
    )
    print("Training complete. Metrics:")
    pprint(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train water-stress models for Tunisia.")
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
