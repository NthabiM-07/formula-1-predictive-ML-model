"""
F1 Race Outcome Prediction - Inference Script
=============================================

This script loads a trained F1 prediction model, reads new race data,
performs necessary preprocessing, and outputs predictions to a CSV file.

Usage:
------
python predict.py

The script expects:
1. A trained model file 'f1_prediction_model.joblib' in the current directory
2. New race data in 'data/new_race_data.csv'

Output:
-------
Predictions will be saved to 'data/predictions.csv'
"""

import pandas as pd
import numpy as np
import os
import joblib
from f1_prediction_model import F1PredictionModel

def ensure_directory_exists(directory):
    """
    Ensure that the specified directory exists.
    
    Parameters:
    -----------
    directory : str
        Directory path to check/create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def load_new_data(file_path):
    """
    Load new race data from a CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing new race data
        
    Returns:
    --------
    DataFrame
        DataFrame containing the new race data
    """
    print(f"Loading new race data from {file_path}...")
    try:
        data = pd.read_csv(file_path)
        print(f"Successfully loaded data with {len(data)} rows and {len(data.columns)} columns")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    """
    Preprocess the new race data to prepare it for prediction.
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame containing the new race data
        
    Returns:
    --------
    DataFrame
        Preprocessed data ready for prediction
    """
    print("Preprocessing data...")
    
    # Make a copy to avoid modifying the original data
    processed_data = data.copy()
    
    # Convert time strings to seconds if needed
    for col in ['Q1 Time', 'Q2 Time', 'Q3 Time']:
        if col in processed_data.columns:
            processed_data[col + '_seconds'] = processed_data[col].apply(
                lambda x: pd.to_timedelta(x).total_seconds() if pd.notna(x) and isinstance(x, str) else x
            )
    
    # Calculate best qualifying time if not already present
    if 'best_qualifying_time' not in processed_data.columns and any(col in processed_data.columns for col in ['Q1 Time_seconds', 'Q2 Time_seconds', 'Q3 Time_seconds']):
        processed_data['best_qualifying_time'] = processed_data.apply(
            lambda row: row.get('Q3 Time_seconds', np.nan) if pd.notna(row.get('Q3 Time_seconds', np.nan)) 
                   else (row.get('Q2 Time_seconds', np.nan) if pd.notna(row.get('Q2 Time_seconds', np.nan)) 
                         else row.get('Q1 Time_seconds', np.nan)),
            axis=1
        )
    
    # Ensure required columns are present
    required_columns = ['Position_x', 'best_qualifying_time', 'GridPosition']
    for col in required_columns:
        if col not in processed_data.columns:
            print(f"Warning: Required column '{col}' not found in data")
    
    # Handle team column naming
    if 'Team' in processed_data.columns and 'Team_x' not in processed_data.columns:
        processed_data['Team_x'] = processed_data['Team']
    
    # Handle missing values
    processed_data = processed_data.fillna(processed_data.mean(numeric_only=True))
    
    print("Data preprocessing complete")
    return processed_data

def main():
    """Main function to run the prediction pipeline"""
    print("F1 Race Outcome Prediction - Inference")
    print("=====================================\n")
    
    # Ensure data directory exists
    data_dir = "data"
    ensure_directory_exists(data_dir)
    
    # Define file paths
    input_file = os.path.join(data_dir, "new_race_data.csv")
    output_file = os.path.join(data_dir, "predictions.csv")
    model_file = "f1_prediction_model.joblib"
    
    # Step 1: Load the pre-trained model
    print(f"Step 1: Loading the pre-trained model from {model_file}...")
    model = F1PredictionModel()
    model.load_model(model_file)
    print("Model loaded successfully!\n")
    
    # Step 2: Load new race data
    new_data = load_new_data(input_file)
    
    # Step 3: Preprocess the data
    processed_data = preprocess_data(new_data)
    
    # Step 4: Make predictions
    print("Making predictions...")
    try:
        predictions = model.predict(processed_data)
        
        # Step 5: Prepare results
        results = pd.DataFrame({
            'PredictedPosition': predictions
        })
        
        # Add driver and team information if available
        if 'Driver' in new_data.columns:
            results['Driver'] = new_data['Driver']
        elif 'DriverId' in new_data.columns:
            results['DriverId'] = new_data['DriverId']
        
        if 'Team_x' in new_data.columns:
            results['Team'] = new_data['Team_x']
        elif 'Team' in new_data.columns:
            results['Team'] = new_data['Team']
        
        # Add original features for reference
        for col in ['Position_x', 'GridPosition']:
            if col in new_data.columns:
                results[col] = new_data[col]
        
        # Sort by predicted position
        results = results.sort_values('PredictedPosition')
        
        # Save predictions to CSV
        results.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
        # Display preview of results
        print("\nPreview of predictions:")
        print(results.head())
        
    except Exception as e:
        print(f"Error making predictions: {e}")
        raise

if __name__ == "__main__":
    main()