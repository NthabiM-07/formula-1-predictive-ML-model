# Formula 1 Race Prediction Model

This project implements a machine learning model to predict Formula 1 race outcomes based on qualifying results, historical performance, and other race-related features.

## Overview

The F1 Race Prediction Model uses historical Formula 1 data to predict the finishing positions of drivers in races. The model analyzes patterns in qualifying performance, grid positions, and previous race results to make predictions about future race outcomes.

## Features

- **Data Processing**: Automatically loads and processes F1 race data from multiple Grand Prix events
- **Feature Engineering**: Creates meaningful features from raw race data
- **Model Training**: Trains a machine learning model (Random Forest or Gradient Boosting) on historical race data
- **Evaluation**: Provides comprehensive evaluation metrics and visualizations
- **Prediction**: Makes predictions for future races based on qualifying results

## Data

The model uses the following data sources:
- Qualifying results
- Race results
- Driver standings
- Lap times

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- streamlit (for interactive app)

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit
   ```

## Usage

### Basic Usage

```python
from f1_prediction_model import F1PredictionModel

# Create model instance
model = F1PredictionModel(data_dir="path/to/data")

# Load data
data = model.load_data()

# Prepare features
X, y, df = model.prepare_features(data)
model.X = X
model.y = y

# Train model
model.train()

# Evaluate model
metrics = model.evaluate()

# Save model
model.save_model("f1_model.joblib")
```

### Making Predictions

```python
# Load a trained model
model = F1PredictionModel()
model.load_model("f1_model.joblib")

# Create input data for prediction
import pandas as pd
new_data = pd.DataFrame({
    'Position_x': [1, 2, 3],  # Qualifying positions
    'best_qualifying_time': [64.0, 64.5, 64.7],  # Qualifying times in seconds
    'GridPosition': [1, 2, 3],  # Grid positions
    'Team': ['McLaren', 'Ferrari', 'Red Bull Racing']  # Teams
})

# Make predictions
predictions = model.predict(new_data)
print(predictions)
```

### Running the Demo

A demonstration script is included to show how to use the model:

```
python run_f1_model.py
```

This will:
1. Load the F1 data
2. Train a prediction model
3. Evaluate the model's performance
4. Make predictions for a hypothetical race
5. Visualize the predictions

## Model Details

The model uses a machine learning algorithm (Random Forest by default) to predict race finishing positions. It considers several features:

- Qualifying position
- Qualifying time
- Grid position
- Previous race results (when available)
- Team information
- Previous championship standings (when available)

## Evaluation

The model is evaluated using several metrics:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

Visualizations are also generated to show:
- Actual vs. Predicted finishing positions
- Feature importance

## New Components

### Inference Script (predict.py)

A standalone script for making predictions on new race data:

```bash
python predict.py
```

This script:
1. Loads the trained model from `f1_prediction_model.joblib`
2. Reads new race data from `data/new_race_data.csv`
3. Performs necessary preprocessing
4. Outputs predictions to `data/predictions.csv`

### Feature Engineering Suggestions (feature_engineering_suggestions.py)

Explore ideas to improve the model's R² score beyond ~0.08:

```bash
python feature_engineering_suggestions.py
```

This script provides three detailed feature engineering suggestions:
1. Incorporating weather and track conditions
2. Creating more sophisticated driver-team historical performance metrics
3. Applying advanced feature transformations and interactions

Each suggestion includes code snippets demonstrating implementation.

### Interactive Streamlit App (streamlit_app.py)

An interactive web application for uploading data and visualizing predictions:

```bash
streamlit run streamlit_app.py
```

The Streamlit app allows you to:
1. Upload new race data
2. Make predictions using the trained model
3. Visualize prediction results and errors with interactive plots
4. Compare actual vs predicted positions

## Customization

You can customize the model by:
- Changing the model type: `model = F1PredictionModel(model_type="gradient_boosting")`
- Adding more features in the `prepare_features` method
- Modifying the hyperparameter grid in the `build_model` method
- Implementing the feature engineering suggestions from `feature_engineering_suggestions.py`

## License

This project is available under the MIT License.
