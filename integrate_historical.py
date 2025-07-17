#!/usr/bin/env python3
"""
F1 Historical Data Integrator
============================
Integrates historical F1 data (1950-2024) with current race data.
"""

import pandas as pd
import numpy as np
import os
from f1_prediction_model import F1PredictionModel

def integrate_historical_data():
    """Main integration function."""
    print("🏎️ F1 Historical Data Integration")
    print("=" * 50)
    
    historical_dir = "f1 metadata 1950 - 2024"
    
    # Check if historical data exists
    if not os.path.exists(historical_dir):
        print(f"❌ Historical data directory not found: {historical_dir}")
        return
    
    print("📚 Loading historical F1 data...")
    
    # Load required historical files
    try:
        races_df = pd.read_csv(os.path.join(historical_dir, "races.csv"))
        results_df = pd.read_csv(os.path.join(historical_dir, "results.csv"))
        qualifying_df = pd.read_csv(os.path.join(historical_dir, "qualifying.csv"))
        constructors_df = pd.read_csv(os.path.join(historical_dir, "constructors.csv"))
        
        print(f"  ✓ races: {len(races_df):,} records")
        print(f"  ✓ results: {len(results_df):,} records")
        print(f"  ✓ qualifying: {len(qualifying_df):,} records")
        print(f"  ✓ constructors: {len(constructors_df):,} records")
        
    except Exception as e:
        print(f"❌ Error loading historical data: {e}")
        return
    
    # Filter to recent years (2018+) for relevance
    start_year = 2018
    print(f"🔍 Filtering data from {start_year} onwards...")
    
    races_df['year'] = pd.to_datetime(races_df['date']).dt.year
    recent_races = races_df[races_df['year'] >= start_year]
    recent_race_ids = set(recent_races['raceId'].values)
    
    print(f"  📅 Found {len(recent_races)} races from {start_year}-2024")
    
    # Filter results and qualifying to recent races
    results_recent = results_df[results_df['raceId'].isin(recent_race_ids)]
    qualifying_recent = qualifying_df[qualifying_df['raceId'].isin(recent_race_ids)]
    
    print(f"  ✓ Filtered results: {len(results_recent):,} records")
    print(f"  ✓ Filtered qualifying: {len(qualifying_recent):,} records")
    
    # Merge qualifying and race results
    print("🔄 Merging qualifying and race results...")
    
    merged_df = pd.merge(
        qualifying_recent,
        results_recent,
        on=['raceId', 'driverId', 'constructorId'],
        how='inner',
        suffixes=('_quali', '_race')
    )
    
    # Add constructor names
    merged_df = pd.merge(
        merged_df,
        constructors_df[['constructorId', 'name']],
        left_on='constructorId',
        right_on='constructorId',
        how='left'
    )
    
    print(f"  ✓ Merged dataset: {len(merged_df):,} records")
    
    # Create features matching current model format
    print("🔧 Creating model features...")
    
    # Parse qualifying times
    def parse_time(time_str):
        if pd.isna(time_str) or time_str == '\\N':
            return float('nan')
        try:
            if ':' in str(time_str):
                parts = str(time_str).split(':')
                return float(parts[0]) * 60 + float(parts[1])
            return float(time_str)
        except:
            return float('nan')
    
    # Create feature dataframe
    feature_df = pd.DataFrame()
    feature_df['Position_x'] = pd.to_numeric(merged_df['position_quali'], errors='coerce')
    feature_df['GridPosition'] = pd.to_numeric(merged_df['grid'], errors='coerce')
    feature_df['Team'] = merged_df['name']
    
    # Use best available qualifying time
    q1_times = merged_df['q1'].apply(parse_time)
    q2_times = merged_df['q2'].apply(parse_time)
    q3_times = merged_df['q3'].apply(parse_time)
    
    feature_df['best_qualifying_time'] = q3_times.fillna(q2_times).fillna(q1_times)
    feature_df['grid_position_diff'] = feature_df['GridPosition'] - feature_df['Position_x']
    
    # Target variable (race finishing position)
    target = pd.to_numeric(merged_df['position_race'], errors='coerce')
    
    # Clean data - remove invalid records
    valid_mask = (
        (feature_df['Position_x'].notna()) & 
        (target.notna()) & 
        (target > 0) & 
        (target <= 25)  # Reasonable finishing positions
    )
    
    feature_df = feature_df[valid_mask]
    target = target[valid_mask]
    
    # Fill missing values
    numeric_cols = feature_df.select_dtypes(include=['number']).columns
    feature_df[numeric_cols] = feature_df[numeric_cols].fillna(feature_df[numeric_cols].median())
    feature_df['Team'] = feature_df['Team'].fillna('Unknown')
    
    print(f"  ✓ Created {len(feature_df)} training samples")
    
    # Load current data
    print("🔄 Loading current race data...")
    
    f1_model = F1PredictionModel(data_dir=".")
    current_data = f1_model.load_data()
    current_X, current_y, _ = f1_model.prepare_features(current_data)
    
    print(f"  📊 Current data: {len(current_X)} samples")
    print(f"  📚 Historical data: {len(feature_df)} samples")
    
    # Find common features
    common_features = list(set(feature_df.columns) & set(current_X.columns))
    print(f"  🔗 Common features: {common_features}")
    
    # Combine datasets
    print("🔄 Combining datasets...")
    
    combined_X = pd.concat([
        feature_df[common_features], 
        current_X[common_features]
    ], ignore_index=True)
    
    combined_y = pd.concat([target, current_y], ignore_index=True)
    
    print(f"  ✅ Combined dataset: {len(combined_X)} samples")
    print(f"  📈 Data increase: {len(combined_X) - len(current_X):+} samples")
    
    # Train enhanced model
    print("🏋️ Training enhanced model...")
    
    enhanced_model = F1PredictionModel(data_dir=".")
    enhanced_model.X = combined_X
    enhanced_model.y = combined_y
    
    enhanced_model.train()
    
    # Get metrics without plots
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np
    
    # Make predictions on training data for metrics
    y_pred = enhanced_model.model.predict(enhanced_model.X)
    
    metrics = {
        'mse': mean_squared_error(enhanced_model.y, y_pred),
        'rmse': np.sqrt(mean_squared_error(enhanced_model.y, y_pred)),
        'mae': mean_absolute_error(enhanced_model.y, y_pred),
        'r2': r2_score(enhanced_model.y, y_pred)
    }
    
    print(f"  🎯 Enhanced Model Performance:")
    print(f"     R² Score: {metrics['r2']:.4f}")
    print(f"     MAE: {metrics['mae']:.2f} positions")
    print(f"     RMSE: {metrics['rmse']:.2f} positions")
    
    # Save enhanced model
    enhanced_model.save_model('f1_prediction_model_enhanced.joblib')
    
    # Save integrated dataset
    print("💾 Saving integrated dataset...")
    
    dataset = combined_X.copy()
    dataset['target_position'] = combined_y
    dataset.to_csv('integrated_f1_dataset.csv', index=False)
    
    print(f"\n🎉 Integration Complete!")
    print(f"  📊 Final dataset: {len(combined_X):,} samples")
    print(f"  🎯 Model performance: R² = {metrics['r2']:.4f}")
    print(f"  💾 Enhanced model: f1_prediction_model_enhanced.joblib")
    print(f"  📁 Dataset: integrated_f1_dataset.csv")
    
    return True

if __name__ == "__main__":
    integrate_historical_data()
