#!/usr/bin/env python
"""
Predict water stress levels for 2030 based on trained models and historical data.
Usage: python scripts/predict_2030.py [--models_dir models_tuned] [--output_dir predictions]
"""
import argparse
import os
import pandas as pd
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict_future import predict_water_stress_2030, save_prediction_report


def main(args):
    """Generate 2030 predictions for all models."""
    models_dir = args.models_dir
    output_dir = args.output_dir
    last_year = args.last_year
    processed_data_path = args.processed
    
    # Load processed data
    if not os.path.exists(processed_data_path):
        print(f"Error: Processed data not found at {processed_data_path}")
        return
    
    processed_data = pd.read_csv(processed_data_path, index_col=0)
    processed_data.index = processed_data.index.astype(int)
    print(f"Loaded {len(processed_data)} years of data (up to {processed_data.index.max()})")
    
    # Find all trained models
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
    if not model_files:
        print(f"Error: No trained models found in {models_dir}")
        return
    
    print(f"Found {len(model_files)} trained models")
    
    # Use the features that were selected during training (after leakage/collinearity filtering)
    # These are the 7 features that models were trained with
    feature_columns = [
        "Annual freshwater withdrawals, domestic (% of total freshwater withdrawal)",
        "Annual freshwater withdrawals, industry (% of total freshwater withdrawal)",
        "Annual freshwater withdrawals, total (% of internal resources)",
        "Renewable internal freshwater resources per capita (cubic meters)",
        "Renewable internal freshwater resources, total (billion cubic meters)",
        "Water productivity, total (constant 2015 US$ GDP per cubic meter of total freshwater withdrawal)",
        "year",
    ]
    
    # Verify all features exist in the data
    missing_features = [c for c in feature_columns if c not in processed_data.columns]
    if missing_features:
        print(f"Warning: Some features are missing from data: {missing_features}")
    
    available_features = [c for c in feature_columns if c in processed_data.columns]
    print(f"Using {len(available_features)} features for 2030 prediction")
    
    # Generate predictions
    predictions = []
    for model_file in sorted(model_files):
        model_path = os.path.join(models_dir, model_file)
        model_name = model_file.replace('.joblib', '')
        
        try:
            pred = predict_water_stress_2030(
                model_path=model_path,
                processed_data=processed_data,
                feature_columns=feature_columns,
                last_year=last_year,
                target_year=2030,
                scenario_method=args.scenario_method,
                include_year_feature=True
            )
            predictions.append(pred)
            print(f"✓ {model_name}: 2030 prediction = {pred['prediction']:.4f}")
        except Exception as e:
            print(f"✗ {model_name}: {e}")
    
    if not predictions:
        print("Error: No successful predictions generated")
        return
    
    # Save report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'water_stress_2030_predictions.csv')
    report_df = save_prediction_report(predictions, report_path)
    
    # Print summary
    print("\n=== 2030 Water Stress Predictions for Tunisia ===")
    print(f"Scenario Method: {args.scenario_method}")
    print(f"Projections from {last_year} to 2030\n")
    print(report_df.to_string(index=False))
    
    # Save detailed report
    detailed_path = os.path.join(output_dir, 'water_stress_2030_detailed.txt')
    with open(detailed_path, 'w', encoding='utf-8') as fh:
        fh.write("=== Tunisia Water Stress 2030 Projections ===\n\n")
        fh.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"Scenario Method: {args.scenario_method}\n")
        fh.write(f"Data Range: {processed_data.index.min()}-{last_year}\n")
        fh.write(f"Prediction Year: 2030\n\n")
        
        for pred in predictions:
            model_name = os.path.basename(pred['model_path']).replace('.joblib', '')
            fh.write(f"Model: {model_name}\n")
            fh.write(f"  Predicted 2030 Water Stress: {pred['prediction']:.6f}\n")
            fh.write(f"  Last Historical Value (2024): {processed_data.loc[last_year, 'Annual freshwater withdrawals, agriculture (% of total freshwater withdrawal)']:.6f}\n\n")
    
    print(f"\nDetailed report saved to: {detailed_path}")
    print(f"Summary CSV saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict water stress for 2030 scenario")
    parser.add_argument("--models_dir", default="models_tuned", help="Directory with trained models")
    parser.add_argument("--processed", default=os.path.join("data", "processed", "processed_tunisia.csv"),
                       help="Path to processed data CSV")
    parser.add_argument("--output_dir", default="predictions", help="Output directory for predictions")
    parser.add_argument("--last_year", type=int, default=2024, help="Last year with historical data")
    parser.add_argument("--scenario_method", default="linear", 
                       choices=["linear", "exponential", "average", "last_value"],
                       help="Method for trend extrapolation")
    
    args = parser.parse_args()
    main(args)
