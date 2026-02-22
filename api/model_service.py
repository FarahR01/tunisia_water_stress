"""Model loading and prediction service."""
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import (
    DEFAULT_MODELS_DIR, 
    DEFAULT_MODEL_NAME, 
    AVAILABLE_MODELS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    PROCESSED_DATA_PATH
)


class ModelService:
    """Service for loading models and making predictions."""
    
    def __init__(self, models_dir: Path = None):
        self.models_dir = models_dir or DEFAULT_MODELS_DIR
        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.scaler: Optional[StandardScaler] = None
        self.historical_data: Optional[pd.DataFrame] = None
        self._load_models()
        self._load_metrics()
        self._load_historical_data()
    
    def _load_models(self):
        """Load all available models from disk."""
        for model_name in AVAILABLE_MODELS:
            model_path = self.models_dir / f"{model_name}.joblib"
            if model_path.exists():
                try:
                    self.models[model_name] = joblib.load(model_path)
                    print(f"✓ Loaded model: {model_name}")
                except Exception as e:
                    print(f"✗ Failed to load {model_name}: {e}")
        
        if not self.models:
            raise RuntimeError(f"No models found in {self.models_dir}")
    
    def _load_metrics(self):
        """Load model metrics from disk."""
        metrics_path = self.models_dir / "metrics.csv"
        if metrics_path.exists():
            try:
                df = pd.read_csv(metrics_path)
                for _, row in df.iterrows():
                    self.metrics[row['model']] = {
                        'MAE': row['MAE'],
                        'RMSE': row['RMSE'],
                        'R2': row['R2']
                    }
            except Exception as e:
                print(f"Warning: Could not load metrics: {e}")
    
    def _load_historical_data(self):
        """Load historical data for trend extrapolation."""
        if PROCESSED_DATA_PATH.exists():
            try:
                self.historical_data = pd.read_csv(PROCESSED_DATA_PATH, index_col=0)
                self.historical_data.index = self.historical_data.index.astype(int)
                print(f"✓ Loaded historical data: {len(self.historical_data)} years")
                
                # Fit scaler on historical data for linear models
                self._fit_scaler()
            except Exception as e:
                print(f"Warning: Could not load historical data: {e}")
    
    def _fit_scaler(self):
        """Fit scaler on historical feature data."""
        if self.historical_data is not None:
            try:
                # Get feature columns that exist in data
                available_features = [c for c in FEATURE_COLUMNS if c in self.historical_data.columns or c == 'year']
                feature_data = self.historical_data.copy()
                
                # Add year column if needed
                if 'year' not in feature_data.columns:
                    feature_data['year'] = feature_data.index
                
                # Select only available features and drop NaN
                feature_cols = [c for c in available_features if c in feature_data.columns]
                if feature_cols:
                    clean_data = feature_data[feature_cols].dropna()
                    if len(clean_data) > 0:
                        self.scaler = StandardScaler()
                        self.scaler.fit(clean_data)
            except Exception as e:
                print(f"Warning: Could not fit scaler: {e}")
    
    def get_available_models(self) -> list:
        """Get list of loaded models."""
        return list(self.models.keys())
    
    def get_model_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get metrics for all models."""
        return self.metrics
    
    def get_best_model_name(self) -> str:
        """Get the name of the best performing model based on R2."""
        if not self.metrics:
            return DEFAULT_MODEL_NAME
        
        best_model = max(self.metrics.items(), key=lambda x: x[1].get('R2', float('-inf')))
        return best_model[0]
    
    def _prepare_features(self, features: Dict[str, float], model_name: str) -> np.ndarray:
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
        water_productivity: Optional[float] = None,
        freshwater_withdrawals_agriculture: Optional[float] = None,
        freshwater_withdrawals_industry: Optional[float] = None,
        renewable_freshwater_resources: Optional[float] = None,
        model_name: Optional[str] = None
    ) -> Tuple[float, str, Dict[str, float]]:
        """
        Make a water stress prediction.
        
        Returns:
            Tuple of (prediction, model_used, input_features)
        """
        # Use best model if not specified
        if model_name is None or model_name not in self.models:
            model_name = self.get_best_model_name()
        
        model = self.models[model_name]
        
        # Get default values from historical data if available
        defaults = self._get_historical_defaults(year)
        
        # Build feature dict
        features = {
            'year': year,
            'water_productivity': water_productivity if water_productivity is not None else defaults.get('water_productivity', 8.0),
            'freshwater_withdrawals_agriculture': freshwater_withdrawals_agriculture if freshwater_withdrawals_agriculture is not None else defaults.get('freshwater_withdrawals_agriculture', 75.0),
            'freshwater_withdrawals_industry': freshwater_withdrawals_industry if freshwater_withdrawals_industry is not None else defaults.get('freshwater_withdrawals_industry', 5.0),
            'renewable_freshwater_resources': renewable_freshwater_resources if renewable_freshwater_resources is not None else defaults.get('renewable_freshwater_resources', 4.195)
        }
        
        # Prepare features
        X = self._prepare_features(features, model_name)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        return prediction, model_name, features
    
    def _get_historical_defaults(self, year: int) -> Dict[str, float]:
        """Get historical values for a given year or extrapolate."""
        defaults = {}
        
        if self.historical_data is None:
            return defaults
        
        try:
            if year in self.historical_data.index:
                row = self.historical_data.loc[year]
                # Map column names to simplified keys
                for col in self.historical_data.columns:
                    if 'water productivity' in col.lower():
                        defaults['water_productivity'] = row[col] if pd.notna(row[col]) else None
                    elif 'agriculture' in col.lower() and 'freshwater' in col.lower():
                        defaults['freshwater_withdrawals_agriculture'] = row[col] if pd.notna(row[col]) else None
                    elif 'industry' in col.lower() and 'freshwater' in col.lower():
                        defaults['freshwater_withdrawals_industry'] = row[col] if pd.notna(row[col]) else None
                    elif 'renewable internal freshwater' in col.lower() and 'total' in col.lower():
                        defaults['renewable_freshwater_resources'] = row[col] if pd.notna(row[col]) else None
            else:
                # Extrapolate from most recent data
                defaults = self._extrapolate_features(year)
        except Exception:
            pass
        
        return defaults
    
    def _extrapolate_features(self, target_year: int, method: str = 'linear') -> Dict[str, float]:
        """Extrapolate features for a future year."""
        extrapolated = {}
        
        if self.historical_data is None:
            return extrapolated
        
        feature_mapping = {
            'water_productivity': 'Water productivity, total (constant 2015 US$ GDP per cubic meter of total freshwater withdrawal)',
            'freshwater_withdrawals_agriculture': 'Annual freshwater withdrawals, agriculture (% of total freshwater withdrawal)',
            'freshwater_withdrawals_industry': 'Annual freshwater withdrawals, industry (% of total freshwater withdrawal)',
            'renewable_freshwater_resources': 'Renewable internal freshwater resources, total (billion cubic meters)'
        }
        
        for key, col in feature_mapping.items():
            if col in self.historical_data.columns:
                col_data = self.historical_data[col].dropna()
                if len(col_data) >= 2:
                    last_year = col_data.index.max()
                    if method == 'linear':
                        recent_years = min(5, len(col_data))
                        values = col_data.values
                        slope = (values[-1] - values[-recent_years]) / recent_years
                        extrapolated[key] = values[-1] + slope * (target_year - last_year)
                    else:
                        extrapolated[key] = col_data.iloc[-1]
        
        return extrapolated
    
    def predict_scenario(
        self, 
        target_year: int = 2030,
        trend_method: str = 'linear',
        model_name: Optional[str] = None
    ) -> Tuple[float, str, Dict[str, float], str]:
        """
        Make a scenario-based prediction for a future year.
        
        Returns:
            Tuple of (prediction, model_used, extrapolated_features, interpretation)
        """
        # Use best model if not specified
        if model_name is None or model_name not in self.models:
            model_name = self.get_best_model_name()
        
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
        """Generate a human-readable interpretation of the prediction."""
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


# Global model service instance
_model_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Get or create the global model service instance."""
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service
