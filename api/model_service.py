"""Model loading and prediction service.

This module provides the ModelService class for:
- Loading trained ML models from disk
- Making predictions with proper feature preparation
- Trend extrapolation for future scenarios
- Model metrics retrieval

The service is designed to be injected via FastAPI dependencies.
"""
import logging
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    VALID_TREND_METHODS,
    get_settings
)
from .exceptions import (
    ModelNotFoundError,
    ModelLoadError,
    DataNotFoundError,
    PredictionError,
    InvalidTrendMethodError
)

logger = logging.getLogger("api.model_service")


class ModelService:
    """
    Service for loading models and making predictions.
    
    This class manages the lifecycle of ML models and provides
    a clean interface for making predictions. It's designed to
    be instantiated once and injected into endpoints via Depends().
    
    Attributes:
        models_dir: Directory containing model files
        data_path: Path to historical data CSV
        models: Dictionary of loaded model objects
        metrics: Dictionary of model performance metrics
    """
    
    def __init__(
        self, 
        models_dir: Path | None = None,
        data_path: Path | None = None
    ):
        """
        Initialize the model service.
        
        Args:
            models_dir: Directory containing .joblib model files
            data_path: Path to processed data CSV for historical lookups
            
        Raises:
            ModelLoadError: If no models could be loaded
        """
        settings = get_settings()
        self.models_dir = models_dir or settings.models_dir
        self.data_path = data_path or settings.processed_data_path
        self._available_model_names = settings.available_models
        
        self.models: dict[str, Any] = {}
        self.metrics: dict[str, dict[str, float]] = {}
        self.scaler: StandardScaler | None = None
        self.historical_data: pd.DataFrame | None = None
        
        self._load_models()
        self._load_metrics()
        self._load_historical_data()
    
    def _load_models(self) -> None:
        """Load all available models from disk."""
        for model_name in self._available_model_names:
            model_path = self.models_dir / f"{model_name}.joblib"
            if model_path.exists():
                try:
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
        
        if not self.models:
            raise ModelLoadError(
                str(self.models_dir),
                "No models found or all failed to load"
            )
    
    def _load_metrics(self) -> None:
        """Load model metrics from disk."""
        metrics_path = self.models_dir / "metrics.csv"
        if metrics_path.exists():
            try:
                df = pd.read_csv(metrics_path)
                for _, row in df.iterrows():
                    self.metrics[row['model']] = {
                        'MAE': float(row['MAE']),
                        'RMSE': float(row['RMSE']),
                        'R2': float(row['R2'])
                    }
                logger.info(f"Loaded metrics for {len(self.metrics)} models")
            except Exception as e:
                logger.warning(f"Could not load metrics: {e}")
    
    def _load_historical_data(self) -> None:
        """Load historical data for trend extrapolation."""
        if self.data_path.exists():
            try:
                self.historical_data = pd.read_csv(self.data_path, index_col=0)
                self.historical_data.index = self.historical_data.index.astype(int)
                logger.info(f"Loaded historical data: {len(self.historical_data)} years")
                self._fit_scaler()
            except Exception as e:
                logger.warning(f"Could not load historical data: {e}")
    
    def _fit_scaler(self) -> None:
        """Fit scaler on historical feature data for linear models."""
        if self.historical_data is None:
            return
            
        try:
            available_features = [
                c for c in FEATURE_COLUMNS 
                if c in self.historical_data.columns or c == 'year'
            ]
            feature_data = self.historical_data.copy()
            
            if 'year' not in feature_data.columns:
                feature_data['year'] = feature_data.index
            
            feature_cols = [c for c in available_features if c in feature_data.columns]
            if feature_cols:
                clean_data = feature_data[feature_cols].dropna()
                if len(clean_data) > 0:
                    self.scaler = StandardScaler()
                    self.scaler.fit(clean_data)
                    logger.debug("Fitted feature scaler")
        except Exception as e:
            logger.warning(f"Could not fit scaler: {e}")
    
    def get_available_models(self) -> list[str]:
        """Get list of loaded model names."""
        return list(self.models.keys())
    
    def get_model_metrics(self) -> dict[str, dict[str, float]]:
        """Get performance metrics for all loaded models."""
        return self.metrics
    
    def get_best_model_name(self) -> str:
        """
        Get the name of the best performing model based on R2 score.
        
        Returns:
            Name of best model, or default if no metrics available
        """
        if not self.metrics:
            settings = get_settings()
            return settings.default_model_name
        
        best_model = max(
            self.metrics.items(), 
            key=lambda x: x[1].get('R2', float('-inf'))
        )
        return best_model[0]
    
    def _validate_model_name(self, model_name: str | None) -> str:
        """
        Validate and resolve model name.
        
        Args:
            model_name: Requested model name or None for best model
            
        Returns:
            Valid model name
            
        Raises:
            ModelNotFoundError: If specified model doesn't exist
        """
        if model_name is None:
            return self.get_best_model_name()
        
        if model_name not in self.models:
            raise ModelNotFoundError(model_name, self.get_available_models())
        
        return model_name
    
    def _prepare_features(
        self, 
        features: dict[str, float], 
        model_name: str
    ) -> np.ndarray:
        """Prepare feature array for prediction."""
        # Ensure features are in the correct order
        feature_values = []
        for col in FEATURE_COLUMNS:
            if col == 'year':
                feature_values.append(features.get('year', 2024))
            elif 'water productivity' in col.lower():
                feature_values.append(features.get('water_productivity', 8.0))
            elif 'agriculture' in col.lower():
                feature_values.append(features.get('freshwater_withdrawals_agriculture', 75.0))
            elif 'industry' in col.lower():
                feature_values.append(features.get('freshwater_withdrawals_industry', 5.0))
            elif 'renewable' in col.lower():
                feature_values.append(features.get('renewable_freshwater_resources', 4.195))
        
        feature_array = np.array(feature_values).reshape(1, -1)
        
        # Scale features for linear models
        if model_name in ('LinearRegression', 'Ridge', 'Lasso') and self.scaler is not None:
            try:
                feature_array = self.scaler.transform(feature_array)
            except Exception:
                pass  # Use unscaled features if scaling fails
        
        return feature_array
    
    def predict(
        self, 
        year: int,
        water_productivity: float | None = None,
        freshwater_withdrawals_agriculture: float | None = None,
        freshwater_withdrawals_industry: float | None = None,
        renewable_freshwater_resources: float | None = None,
        model_name: str | None = None
    ) -> tuple[float, str, dict[str, float]]:
        """
        Make a water stress prediction.
        
        Args:
            year: Year for prediction
            water_productivity: Optional water productivity value
            freshwater_withdrawals_agriculture: Optional agriculture withdrawal %
            freshwater_withdrawals_industry: Optional industry withdrawal %
            renewable_freshwater_resources: Optional renewable resources value
            model_name: Model to use (uses best if None)
            
        Returns:
            Tuple of (prediction, model_used, input_features)
            
        Raises:
            ModelNotFoundError: If specified model doesn't exist
            PredictionError: If prediction fails
        """
        # Validate and resolve model
        model_name = self._validate_model_name(model_name)
        model = self.models[model_name]
        
        # Get default values from historical data if available
        defaults = self._get_historical_defaults(year)
        
        # Build feature dict with fallbacks
        features = {
            'year': year,
            'water_productivity': (
                water_productivity if water_productivity is not None 
                else defaults.get('water_productivity', 8.0)
            ),
            'freshwater_withdrawals_agriculture': (
                freshwater_withdrawals_agriculture if freshwater_withdrawals_agriculture is not None 
                else defaults.get('freshwater_withdrawals_agriculture', 75.0)
            ),
            'freshwater_withdrawals_industry': (
                freshwater_withdrawals_industry if freshwater_withdrawals_industry is not None 
                else defaults.get('freshwater_withdrawals_industry', 5.0)
            ),
            'renewable_freshwater_resources': (
                renewable_freshwater_resources if renewable_freshwater_resources is not None 
                else defaults.get('renewable_freshwater_resources', 4.195)
            )
        }
        
        try:
            # Prepare features
            X = self._prepare_features(features, model_name)
            
            # Make prediction
            prediction = float(model.predict(X)[0])
            
            logger.debug(f"Prediction for {year}: {prediction:.2f}% using {model_name}")
            return prediction, model_name, features
            
        except Exception as e:
            logger.error(f"Prediction failed for year {year}: {e}")
            raise PredictionError(year, str(e))
    
    def _get_historical_defaults(self, year: int) -> dict[str, float]:
        """
        Get historical values for a given year or extrapolate.
        
        Args:
            year: Year to get defaults for
            
        Returns:
            Dictionary of feature defaults
        """
        defaults: dict[str, float] = {}
        
        if self.historical_data is None:
            return defaults
        
        try:
            if year in self.historical_data.index:
                row = self.historical_data.loc[year]
                # Map column names to simplified keys
                for col in self.historical_data.columns:
                    col_lower = col.lower()
                    if 'water productivity' in col_lower:
                        if pd.notna(row[col]):
                            defaults['water_productivity'] = float(row[col])
                    elif 'agriculture' in col_lower and 'freshwater' in col_lower:
                        if pd.notna(row[col]):
                            defaults['freshwater_withdrawals_agriculture'] = float(row[col])
                    elif 'industry' in col_lower and 'freshwater' in col_lower:
                        if pd.notna(row[col]):
                            defaults['freshwater_withdrawals_industry'] = float(row[col])
                    elif 'renewable internal freshwater' in col_lower and 'total' in col_lower:
                        if pd.notna(row[col]):
                            defaults['renewable_freshwater_resources'] = float(row[col])
            else:
                # Extrapolate from most recent data
                defaults = self._extrapolate_features(year, method='linear')
        except Exception as e:
            logger.debug(f"Could not get historical defaults for {year}: {e}")
        
        return defaults
    
    def _extrapolate_features(
        self, 
        target_year: int, 
        method: Literal["linear", "exponential", "last_value", "average"] = 'linear'
    ) -> dict[str, float]:
        """
        Extrapolate features for a future year.
        
        Args:
            target_year: Year to extrapolate to
            method: Extrapolation method
            
        Returns:
            Dictionary of extrapolated feature values
        """
        extrapolated: dict[str, float] = {}
        
        if self.historical_data is None:
            return extrapolated
        
        feature_mapping = {
            'water_productivity': 'Water productivity, total (constant 2015 US$ GDP per cubic meter of total freshwater withdrawal)',
            'freshwater_withdrawals_agriculture': 'Annual freshwater withdrawals, agriculture (% of total freshwater withdrawal)',
            'freshwater_withdrawals_industry': 'Annual freshwater withdrawals, industry (% of total freshwater withdrawal)',
            'renewable_freshwater_resources': 'Renewable internal freshwater resources, total (billion cubic meters)'
        }
        
        for key, col in feature_mapping.items():
            if col not in self.historical_data.columns:
                continue
                
            col_data = self.historical_data[col].dropna()
            if len(col_data) < 2:
                continue
                
            last_year = int(col_data.index.max())
            values = col_data.values
            
            if method == 'linear':
                recent_years = min(5, len(col_data))
                slope = (values[-1] - values[-recent_years]) / recent_years
                extrapolated[key] = float(values[-1] + slope * (target_year - last_year))
            elif method == 'last_value':
                extrapolated[key] = float(values[-1])
            elif method == 'average':
                extrapolated[key] = float(np.mean(values))
            elif method == 'exponential':
                # Simple exponential: use growth rate from last 5 years
                recent_years = min(5, len(col_data))
                if values[-recent_years] > 0:
                    growth_rate = (values[-1] / values[-recent_years]) ** (1 / recent_years)
                    extrapolated[key] = float(values[-1] * (growth_rate ** (target_year - last_year)))
                else:
                    extrapolated[key] = float(values[-1])
        
        return extrapolated
    
    def predict_scenario(
        self, 
        target_year: int = 2030,
        trend_method: Literal["linear", "exponential", "last_value", "average"] = 'linear',
        model_name: str | None = None
    ) -> tuple[float, str, dict[str, float], str]:
        """
        Make a scenario-based prediction for a future year.
        
        Args:
            target_year: Year to predict for
            trend_method: Method for feature extrapolation
            model_name: Model to use (uses best if None)
            
        Returns:
            Tuple of (prediction, model_used, extrapolated_features, interpretation)
            
        Raises:
            InvalidTrendMethodError: If trend method is invalid
            ModelNotFoundError: If specified model doesn't exist
            PredictionError: If prediction fails
        """
        # Validate trend method
        if trend_method not in VALID_TREND_METHODS:
            raise InvalidTrendMethodError(trend_method)
        
        # Validate model
        model_name = self._validate_model_name(model_name)
        
        # Extrapolate features
        extrapolated = self._extrapolate_features(target_year, trend_method)
        extrapolated['year'] = target_year
        
        # Make prediction
        prediction, model_used, features = self.predict(
            year=target_year,
            water_productivity=extrapolated.get('water_productivity'),
            freshwater_withdrawals_agriculture=extrapolated.get('freshwater_withdrawals_agriculture'),
            freshwater_withdrawals_industry=extrapolated.get('freshwater_withdrawals_industry'),
            renewable_freshwater_resources=extrapolated.get('renewable_freshwater_resources'),
            model_name=model_name
        )
        
        # Generate interpretation
        interpretation = self._generate_interpretation(prediction, target_year)
        
        return prediction, model_used, features, interpretation
    
    def _generate_interpretation(self, prediction: float, year: int) -> str:
        """
        Generate a human-readable interpretation of the prediction.
        
        Args:
            prediction: Predicted water stress percentage
            year: Target year
            
        Returns:
            Interpretation string
        """
        if prediction < 25:
            stress_level = "low"
            outlook = "sustainable water availability"
        elif prediction < 50:
            stress_level = "low-medium"
            outlook = "manageable water situation with monitoring needed"
        elif prediction < 75:
            stress_level = "medium-high"
            outlook = "potential water scarcity concerns"
        elif prediction < 100:
            stress_level = "high"
            outlook = "significant water stress requiring intervention"
        else:
            stress_level = "critical"
            outlook = "severe water scarcity - unsustainable situation"
        
        return (
            f"By {year}, Tunisia is predicted to have {stress_level} water stress "
            f"at {prediction:.1f}%. This indicates {outlook}. "
            f"Note: This prediction is based on historical trend extrapolation "
            f"and assumes continuation of current patterns."
        )


# Note: ModelService instances are managed by the dependencies module.
# Use api.dependencies.get_model_service() for dependency injection.
