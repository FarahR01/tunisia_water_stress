"""MLflow experiment tracking utilities for reproducible ML.

This module provides a wrapper around MLflow for logging experiments with consistent
parameters, metrics, models, and artifacts. It ensures that all experiment runs are
tracked and comparable, enabling reproducible research.

Usage:
    from mlflow_utils import MLflowTracker
    from config_train import TrainConfig

    config = TrainConfig.from_yaml("config/train_config.yaml")
    tracker = MLflowTracker.from_config(config.mlflow)

    with tracker.track_experiment(config=config) as mlflow_logger:
        # Train models
        model = train_model(...)
        metrics = evaluate_model(model)

        # Log automatically
        mlflow_logger.log_metrics(metrics)
        mlflow_logger.log_model(model)
"""

import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from config_train import MLflowConfig

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None


class NoOpMLflowLogger:
    """No-op logger when MLflow is disabled or not installed."""

    def log_param(self, key: str, value: Any) -> None:
        pass

    def log_metric(self, key: str, value: float, step: int = 0) -> None:
        pass

    def log_params(self, params: Dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: Dict[str, float], step: int = 0) -> None:
        pass

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        pass

    def log_table(self, data: Dict[str, Any], key: str) -> None:
        pass

    def log_model(self, model: Any, artifact_path: str) -> None:
        pass


class MLflowLogger:
    """Logger wrapper for MLflow experiment tracking.

    Attributes:
        run_id: MLflow run ID for this experiment session
        tracking_uri: MLflow tracking URI
    """

    def __init__(self, tracking_uri: str):
        """Initialize MLflow logger.

        Args:
            tracking_uri: MLflow tracking server URI or local path
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is not installed. Install with: pip install mlflow")

        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None

    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter.

        Args:
            key: Parameter name
            value: Parameter value (will be converted to string)
        """
        if mlflow.active_run():
            mlflow.log_param(key, str(value))

    def log_metric(self, key: str, value: float, step: int = 0) -> None:
        """Log a single metric.

        Args:
            key: Metric name
            value: Metric value (must be numeric)
            step: Training step or epoch (for time-series metrics)
        """
        if mlflow.active_run():
            mlflow.log_metric(key, float(value), step=step)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters at once.

        Args:
            params: Dictionary of parameter name -> value pairs
        """
        if mlflow.active_run():
            for key, value in params.items():
                mlflow.log_param(key, str(value))

    def log_metrics(self, metrics: Dict[str, float], step: int = 0) -> None:
        """Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric name -> value pairs
            step: Training step (for time-series metrics)
        """
        if mlflow.active_run():
            for key, value in metrics.items():
                mlflow.log_metric(key, float(value), step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log a local file as an artifact.

        Args:
            local_path: Path to local file to upload
            artifact_path: Destination path in MLflow artifact store (optional)
        """
        if mlflow.active_run() and os.path.exists(local_path):
            if os.path.isfile(local_path):
                mlflow.log_artifact(local_path, artifact_path=artifact_path)
            else:
                mlflow.log_artifacts(local_path, artifact_path=artifact_path)

    def log_table(self, data: Dict[str, Any], key: str) -> None:
        """Log a table/dictionary as a JSON artifact.

        Args:
            data: Dictionary to log as table
            key: Name of the artifact file (without .json)
        """
        if mlflow.active_run():
            with mlflow.start_span(name="log_table"):
                # Save to temporary file and log
                temp_path = f"/tmp/{key}.json"
                with open(temp_path, "w") as f:
                    json.dump(data, f, indent=2)
                mlflow.log_artifact(temp_path, artifact_path="tables")
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    def log_config(self, config_dict: Dict[str, Any]) -> None:
        """Log entire configuration as parameters.

        Args:
            config_dict: Configuration dictionary (typically from Pydantic model)
        """
        if mlflow.active_run():

            def flatten_dict(d, parent_key=""):
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}.{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key).items())
                    else:
                        items.append((new_key, v))
                return dict(items)

            flat_config = flatten_dict(config_dict)
            for key, value in flat_config.items():
                mlflow.log_param(key, str(value))

    def get_run_id(self) -> Optional[str]:
        """Get current run ID.

        Returns:
            MLflow run ID or None if no active run
        """
        if mlflow.active_run():
            return mlflow.active_run().info.run_id
        return None


class MLflowTracker:
    """Context manager for MLflow experiment tracking.

    Usage:
        tracker = MLflowTracker(
            tracking_uri="./mlruns",
            experiment_name="my_experiment"
        )

        with tracker.track_experiment(config=config) as logger:
            # Train and log
            model = train(config)
            metrics = evaluate(model)
            logger.log_metrics(metrics)
    """

    def __init__(self, tracking_uri: str, experiment_name: str, enabled: bool = True):
        """Initialize MLflow tracker.

        Args:
            tracking_uri: MLflow tracking server URI or local path
            experiment_name: Name of the experiment for grouping runs
            enabled: Enable MLflow tracking (can be disabled for debugging)
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.enabled = enabled and MLFLOW_AVAILABLE
        self._logger: Optional[MLflowLogger] = None

    @classmethod
    def from_config(cls, mlflow_config: "MLflowConfig") -> "MLflowTracker":
        """Create MLflowTracker from configuration object.

        Args:
            mlflow_config: MLflowConfig Pydantic model instance

        Returns:
            Initialized MLflowTracker
        """
        return cls(
            tracking_uri=mlflow_config.tracking_uri,
            experiment_name=mlflow_config.experiment_name,
            enabled=mlflow_config.enabled,
        )

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """Start a new MLflow run.

        Args:
            run_name: Optional name for this run (timestamp used if None)
            tags: Optional dictionary of tags for the run
        """
        if not self.enabled:
            return

        if not MLFLOW_AVAILABLE:
            print("Warning: MLflow not available, tracking disabled")
            return

        # Set experiment
        mlflow.set_experiment(self.experiment_name)

        # Start run
        if not run_name:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        mlflow.start_run(run_name=run_name)

        if tags:
            mlflow.set_tags(tags)

        self._logger = MLflowLogger(self.tracking_uri)

    def end_run(self):
        """End the current MLflow run."""
        if self.enabled and MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.end_run()
        self._logger = None

    def get_logger(self) -> Any:
        """Get current logger (MLflowLogger or NoOpMLflowLogger)."""
        if self._logger is None:
            if not self.enabled:
                return NoOpMLflowLogger()
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        return self._logger

    def __enter__(self):
        """Context manager entry."""
        self.start_run(
            tags={
                "framework": "scikit-learn",
                "experiment": self.experiment_name,
            }
        )
        return self.get_logger() if self.enabled else NoOpMLflowLogger()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            print(f"Error during run: {exc_val}")
        self.end_run()


class ExperimentAnalyzer:
    """Analyze and compare completed MLflow experiments.

    Usage:
        analyzer = ExperimentAnalyzer(tracking_uri="./mlruns")
        runs = analyzer.get_experiment_runs("water_stress_regression")
        best_run = analyzer.get_best_run(runs, metric_name="test_r2")
    """

    def __init__(self, tracking_uri: str):
        """Initialize analyzer.

        Args:
            tracking_uri: MLflow tracking server URI or local path
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is not installed")

        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)

    def get_experiment_runs(self, experiment_name: str) -> list:
        """Get all runs for an experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            List of MLflow run objects
        """
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                return []
            return mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        except Exception as e:
            print(f"Error fetching runs: {e}")
            return []

    def get_best_run(self, runs, metric_name: str, mode: str = "max"):
        """Find best run based on metric.

        Args:
            runs: List of MLflow run objects or DataFrame from search_runs
            metric_name: Name of metric to optimize
            mode: "max" to maximize or "min" to minimize

        Returns:
            Best run or None
        """
        if runs is None or len(runs) == 0:
            return None

        # Handle DataFrame from search_runs
        if hasattr(runs, "itertuples"):
            runs = list(runs.itertuples(index=False))

        metric_col = f"metrics.{metric_name}"
        best_run = None
        best_value = None

        for run in runs:
            value = getattr(run, metric_col, None)
            if value is not None:
                if (
                    best_value is None
                    or (mode == "max" and value > best_value)
                    or (mode == "min" and value < best_value)
                ):
                    best_value = value
                    best_run = run

        return best_run
