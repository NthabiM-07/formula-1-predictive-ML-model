"""
Quick Model Testing Script
=========================

Simple script to quickly test your F1 model predictions
on new data or validate existing performance.
"""

import pandas as pd
import numpy as np
from f1_prediction_model import F1PredictionModel
import joblib

def quick_test():
    """Quick test of the model with sample data."""
    
    print("🏎️ F1 Model Quick Test")
    print("=" * 30)
    
    # Try to load existing model
    try:
        model = joblib.load('f1_prediction_model.joblib')
        print("✓ Loaded existing model")
    except FileNotFoundError:
        print("No saved model found. Training new model...")
        f1_model = F1PredictionModel(data_dir=".")
        data = f1_model.load_data()
        X, y, df = f1_model.prepare_features(data)
        f1_model.X = X
        f1_model.y = y
        f1_model.train()
        model = f1_model.best_model
        f1_model.save_model()
        print("✓ New model trained and saved")
    
    # Create test data for next race
    print("\n📊 Testing with sample race data...")
    
    test_data = pd.DataFrame({
        'Position_x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Qualifying positions
        'best_qualifying_time': [64.0, 64.3, 64.5, 64.7, 64.9, 
                                65.1, 65.3, 65.5, 65.7, 65.9],
        'GridPosition': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'grid_position_diff': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Team': ['McLaren', 'Ferrari', 'Red Bull Racing', 'Mercedes', 'Aston Martin',
                 'Alpine', 'Williams', 'Haas F1 Team', 'Kick Sauber', 'Racing Bulls']
    })
    
    try:
        # Make predictions
        predictions = model.predict(test_data)
        
        # Display results
        results = pd.DataFrame({
            'Driver': [f'Driver {i+1}' for i in range(len(test_data))],
            'Team': test_data['Team'],
            'Quali_Pos': test_data['Position_x'],
            'Predicted_Finish': predictions.round(1)
        })
        
        print("\n🏁 Race Predictions:")
        print(results.to_string(index=False))
        
        # Simple analysis
        print(f"\n📈 Quick Analysis:")
        print(f"  • Average predicted movement: {np.mean(predictions - test_data['Position_x']):.1f} positions")
        print(f"  • Biggest gainer: Driver {np.argmin(predictions - test_data['Position_x']) + 1}")
        print(f"  • Biggest loser: Driver {np.argmax(predictions - test_data['Position_x']) + 1}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        print("This might be due to feature mismatch or model issues")
        return False

def validate_on_existing_data():
    """Validate model performance on existing data."""
    
    print("\n🔍 Validating on existing data...")
    
    # Load model and data
    f1_model = F1PredictionModel(data_dir=".")
    data = f1_model.load_data()
    X, y, df = f1_model.prepare_features(data)
    
    model = joblib.load('f1_prediction_model.joblib')
    predictions = model.predict(X)
    
    # Calculate simple metrics
    from sklearn.metrics import mean_absolute_error, r2_score
    
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    print(f"  • Mean Absolute Error: {mae:.2f} positions")
    print(f"  • R² Score: {r2:.4f}")
    print(f"  • Dataset size: {len(y)} samples")
    
    # Show some examples
    comparison = pd.DataFrame({
        'Actual': y.values,
        'Predicted': predictions.round(1),
        'Difference': (predictions - y.values).round(1)
    })
    
    print(f"\n📋 Sample Predictions (first 10):")
    print(comparison.head(10).to_string())

if __name__ == "__main__":
    if quick_test():
        validate_on_existing_data()
    print(f"\n✅ Quick test completed!")
