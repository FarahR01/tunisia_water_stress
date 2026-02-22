"""Scenario prediction and forecasting utilities."""
import os
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from sklearn.preprocessing import StandardScaler


def load_trained_model(model_path: str):
    """Load a joblib-saved model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def create_2030_scenario(processed_data: pd.DataFrame, last_year: int = 2024, target_year: int = 2030, 
                         trend_method: str = "linear"):
    """
    Create a 2030 scenario by extrapolating trends from historical data.
    
    Args:
        processed_data: DataFrame with years as index and indicators as columns
        last_year: Last year with actual data
        target_year: Year to predict (default 2030)
        trend_method: 'linear', 'exponential', 'last_value', or 'average'
    
    Returns:
        DataFrame with scenario for target_year
    """
    # Get data up to last_year
    available_data = processed_data[processed_data.index <= last_year].copy()
    
    if len(available_data) < 2:
        raise ValueError("Not enough data to extrapolate trends")
    
    scenario_rows = []
    
    for year in range(last_year + 1, target_year + 1):
        scenario_row = {}
        
        for col in available_data.columns:
            col_data = available_data[col].dropna()
            
            if len(col_data) < 2 or col_data.std() == 0:
                # If insufficient data or no variation, use last value
                scenario_row[col] = col_data.iloc[-1] if len(col_data) > 0 else np.nan
            elif trend_method == "linear":
                # Linear extrapolation
                years_arr = np.array(col_data.index).reshape(-1, 1)
                values_arr = col_data.values
                
                # Simple linear trend: fit to last 5 years if possible
                recent_years = min(5, len(col_data))
                slope = (values_arr[-1] - values_arr[-recent_years]) / recent_years
                scenario_row[col] = values_arr[-1] + slope * (year - last_year)
                
            elif trend_method == "exponential":
                # Exponential growth (or decay)
                values_arr = col_data.values
                if all(v > 0 for v in values_arr[-2:]):
                    growth_rate = values_arr[-1] / values_arr[-2] - 1
                    scenario_row[col] = values_arr[-1] * ((1 + growth_rate) ** (year - last_year))
                else:
                    scenario_row[col] = values_arr[-1]
                    
            elif trend_method == "average":
                # Use average value
                scenario_row[col] = col_data.mean()
            else:  # last_value
                scenario_row[col] = col_data.iloc[-1]
        
        # Add year column
        scenario_row_df = pd.DataFrame(scenario_row, index=[year])
        scenario_rows.append(scenario_row_df)
    
    scenario_2030 = pd.concat(scenario_rows, axis=0)
    scenario_2030.index.name = 'Year'
    
    return scenario_2030


def prepare_features_for_prediction(feature_data: pd.DataFrame, feature_columns: list, target_year: int = None) -> pd.DataFrame:
    """
    Prepare feature matrix for model prediction.
    
    Args:
        feature_data: DataFrame with features (index = year)
        feature_columns: List of feature names expected by the model
        target_year: Year to add to feature data (if 'year' is in feature_columns)
    
    Returns:
        DataFrame with selected features in correct order
    """
    # Create a copy to avoid modifying original
    data = feature_data.copy()
    
    # Handle 'year' feature - add it if requested and not present
    if 'year' in feature_columns and 'year' not in data.columns:
        if target_year is not None:
            # For 2D arrays (single row), this is the 2030 scenario
            if isinstance(data.index, int) or len(data) == 1:
                data['year'] = target_year
            else:
                # For historical data, use index as year
                data['year'] = data.index
    
    available_cols = [c for c in feature_columns if c in data.columns]
    missing_cols = [c for c in feature_columns if c not in data.columns]
    
    if missing_cols:
        print(f"Warning: Missing features: {missing_cols}")
    
    return data[available_cols]


def predict_water_stress_2030(model_path: str, processed_data: pd.DataFrame, 
                               feature_columns: list, last_year: int = 2024,
                               target_year: int = 2030, scenario_method: str = "linear",
                               include_year_feature: bool = True) -> dict:
    """
    Predict water stress for 2030 given a trained model.
    
    Args:
        model_path: Path to saved model (joblib)
        processed_data: Historical processed data
        feature_columns: List of feature names used in training
        last_year: Last year with data
        target_year: Year to predict
        scenario_method: How to extrapolate features ('linear', 'exponential', etc.)
        include_year_feature: If True, add year as a feature (if 'year' in feature_columns)
    
    Returns:
        dict with 'prediction', 'year', 'model_path', 'scenario', 'features'
    """
    # Load model
    model = load_trained_model(model_path)
    
    # Create scenario for 2030
    scenario = create_2030_scenario(processed_data, last_year=last_year, 
                                    target_year=target_year, trend_method=scenario_method)
    
    # Get features for 2030 (including year if requested)
    features_2030 = prepare_features_for_prediction(scenario, feature_columns, target_year=target_year)
    
    # Ensure correct column order (match what the model expects)
    final_features = features_2030[[c for c in feature_columns if c in features_2030.columns]]
    
    # Make prediction
    prediction = model.predict(final_features)[0] if len(final_features) > 0 else np.nan
    
    return {
        'prediction': float(prediction),
        'year': target_year,
        'model_path': model_path,
        'features_used': final_features.to_dict('records')[0] if len(final_features) > 0 else {},
        'scenario_method': scenario_method,
        'last_historical_year': last_year,
    }


def save_prediction_report(predictions: list, output_path: str):
    """
    Save prediction results to CSV.
    
    Args:
        predictions: List of prediction dicts from predict_water_stress_2030
        output_path: Where to save the report
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    report_data = []
    for pred in predictions:
        report_data.append({
            'model': os.path.basename(pred['model_path']).replace('.joblib', ''),
            'prediction_2030': pred['prediction'],
            'scenario_method': pred['scenario_method'],
        })
    
    df = pd.DataFrame(report_data)
    df.to_csv(output_path, index=False)
    print(f"Prediction report saved to: {output_path}")
    
    return df
