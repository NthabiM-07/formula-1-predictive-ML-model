# How to Use the Model to Predict F1 Race Outcomes

This guide provides simple step-by-step instructions for using the F1 prediction model to predict race outcomes.

## Quick Start Guide

### Option 1: Run the Example Script (Easiest)

1. Open a command prompt or terminal
2. Navigate to the project directory
3. Run the example script:
   ```
   python predict_f1_races.py
   ```
4. The script will show example predictions and explain how to use the model

### Option 2: Use the Model in Your Own Code

#### Step 1: Load the Model

```python
from f1_prediction_model import F1PredictionModel

# Create a model instance
model = F1PredictionModel()

# Load the pre-trained model
model.load_model('f1_prediction_model.joblib')
```

#### Step 2: Prepare Your Data

Create a pandas DataFrame with the following columns:

```python
import pandas as pd

# Example data for 5 drivers
data = {
    'Position_x': [1, 2, 3, 4, 5],  # Qualifying positions
    'best_qualifying_time': [92.5, 92.7, 92.9, 93.1, 93.3],  # Qualifying times in seconds
    'GridPosition': [1, 2, 3, 4, 5],  # Grid positions
    'Team_x': ['Red Bull Racing', 'Mercedes', 'Ferrari', 'Red Bull Racing', 'Ferrari']  # Teams
}

# Create DataFrame
prediction_data = pd.DataFrame(data)
```

#### Step 3: Make Predictions

```python
# Get predictions
predictions = model.predict(prediction_data)

# Create a results DataFrame
results = pd.DataFrame({
    'Driver': ['VER', 'HAM', 'LEC', 'PER', 'SAI'],  # Your driver codes
    'Team': prediction_data['Team_x'],
    'Qualifying Position': prediction_data['Position_x'],
    'Predicted Finishing Position': predictions
})

# Sort by predicted position
results = results.sort_values('Predicted Finishing Position')
print(results)
```

## Where to Get the Data

To make predictions for real races, you need:

1. **Qualifying positions**: The position each driver achieved in qualifying
2. **Qualifying times**: The best lap time (in seconds) for each driver in qualifying
3. **Grid positions**: The starting position on the grid (may differ from qualifying due to penalties)
4. **Team information**: The team/constructor for each driver

You can get this information from:
- Official F1 website (formula1.com)
- F1 news websites
- Sports data providers

## Tips for Better Predictions

1. Include as many of the optional features as possible:
   - Previous race positions
   - Previous qualifying positions
   - Championship standings before the race
   
2. Make sure your data format matches what the model expects:
   - Use the exact column names shown above
   - Convert qualifying times to seconds
   - Use consistent team names

3. Remember that predictions are estimates:
   - The model predicts based on historical patterns
   - Unexpected events (weather, crashes, etc.) can affect actual results

## Need More Help?

- Check the full README.md file for more detailed information
- Examine the `predict_f1_races.py` script for a working example
- Look at the `f1_prediction_model.py` file to understand how the model works