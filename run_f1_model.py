import os
import pandas as pd
import matplotlib.pyplot as plt
from f1_prediction_model import F1PredictionModel

def main():
    """
    Demonstrate the usage of the F1PredictionModel class.
    This script loads F1 data, trains a model, and makes predictions.
    """
    print("F1 Race Prediction Model Demo")
    print("=============================")
    
    # Create an instance of the F1PredictionModel
    model = F1PredictionModel(data_dir=".")
    
    # Load data
    print("\nStep 1: Loading data...")
    data = model.load_data()
    
    # Print summary of loaded data
    print("\nData summary:")
    for key, df in data.items():
        if not df.empty:
            print(f"  - {key}: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Prepare features
    print("\nStep 2: Preparing features...")
    X, y, df = model.prepare_features(data)
    
    # Print feature information
    print("\nFeatures used for prediction:")
    for col in X.columns:
        print(f"  - {col}")
    
    # Set X and y in the model
    model.X = X
    model.y = y
    
    # Train the model
    print("\nStep 3: Training model...")
    model.train()
    
    # Evaluate the model
    print("\nStep 4: Evaluating model...")
    metrics = model.evaluate()
    
    # Save the model
    print("\nStep 5: Saving model...")
    model_path = model.save_model()
    
    # Make predictions for the next race
    print("\nStep 6: Making predictions for a hypothetical next race...")
    
    # Create a sample input for prediction
    # This would typically be data from the upcoming race's qualifying
    sample_input = pd.DataFrame({
        'Position_x': [1, 2, 3, 4, 5],  # Qualifying positions
        'best_qualifying_time': [64.0, 64.5, 64.7, 64.9, 65.1],  # Qualifying times in seconds
        'GridPosition': [1, 2, 3, 4, 5],  # Grid positions
        'Team': ['McLaren', 'Ferrari', 'McLaren', 'Ferrari', 'Mercedes']  # Teams
    })
    
    # Fill any missing values that might be expected by the model
    for col in X.columns:
        if col not in sample_input.columns and col not in ['Team_x', 'prev_race_position', 
                                                          'prev_qualifying_position', 
                                                          'Position_prev_standings',
                                                          'Points_prev_standings',
                                                          'grid_position_diff']:
            sample_input[col] = X[col].mean()
    
    # Make predictions
    try:
        predictions = model.predict(sample_input)
        
        # Display predictions
        print("\nPredicted finishing positions:")
        results = pd.DataFrame({
            'Qualifying Position': sample_input['Position_x'],
            'Team': sample_input['Team'],
            'Predicted Finishing Position': predictions.round(1)
        })
        print(results)
        
        # Plot predictions
        plt.figure(figsize=(10, 6))
        plt.bar(results['Qualifying Position'], results['Predicted Finishing Position'], 
                color='skyblue', alpha=0.7)
        plt.plot(results['Qualifying Position'], results['Qualifying Position'], 'r--', 
                 label='Qualifying = Finishing')
        plt.xlabel('Qualifying Position')
        plt.ylabel('Predicted Finishing Position')
        plt.title('Predicted Race Outcomes')
        plt.xticks(results['Qualifying Position'])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('predicted_positions.png')
        print("\nPrediction visualization saved to 'predicted_positions.png'")
        
    except Exception as e:
        print(f"\nError making predictions: {e}")
        print("This could be due to insufficient training data or model compatibility issues.")
    
    print("\nDemo complete! You can now use the F1PredictionModel for your own predictions.")
    print(f"The trained model has been saved to '{model_path}'")

if __name__ == "__main__":
    main()