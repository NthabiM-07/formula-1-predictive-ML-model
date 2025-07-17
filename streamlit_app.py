"""
F1 Race Outcome Prediction - Streamlit App
==========================================

This Streamlit app allows users to:
1. Upload new race data
2. Make predictions using the trained model
3. Visualize prediction results and errors
4. Compare actual vs predicted positions

Usage:
------
streamlit run streamlit_app.py

Requirements:
------------
- streamlit
- pandas
- numpy
- matplotlib
- joblib
- scikit-learn
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from f1_prediction_model import F1PredictionModel

# Set page configuration
st.set_page_config(
    page_title="F1 Race Outcome Prediction",
    page_icon="🏎️",
    layout="wide"
)

@st.cache_data
def load_model(model_path):
    """
    Load the trained model from a file.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model file
        
    Returns:
    --------
    F1PredictionModel
        Loaded model instance
    """
    model = F1PredictionModel()
    model.load_model(model_path)
    return model

def preprocess_data(data):
    """
    Preprocess the uploaded data for prediction.
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame containing the race data
        
    Returns:
    --------
    DataFrame
        Preprocessed data ready for prediction
    """
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
    
    # Handle team column naming
    if 'Team' in processed_data.columns and 'Team_x' not in processed_data.columns:
        processed_data['Team_x'] = processed_data['Team']
    
    # Handle missing values
    processed_data = processed_data.fillna(processed_data.mean(numeric_only=True))
    
    return processed_data

def plot_prediction_errors(actual, predicted):
    """
    Plot a histogram of prediction errors.
    
    Parameters:
    -----------
    actual : array-like
        Actual positions
    predicted : array-like
        Predicted positions
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the histogram
    """
    errors = np.array(actual) - np.array(predicted)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errors, bins=range(int(min(errors))-1, int(max(errors))+2), alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Error')
    
    ax.set_xlabel('Prediction Error (Actual - Predicted)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Prediction Errors')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig

def plot_actual_vs_predicted(actual, predicted, driver_names=None):
    """
    Plot actual vs predicted positions.
    
    Parameters:
    -----------
    actual : array-like
        Actual positions
    predicted : array-like
        Predicted positions
    driver_names : array-like, optional
        Names of drivers for annotations
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the scatter plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    scatter = ax.scatter(actual, predicted, alpha=0.7, s=100)
    
    # Add diagonal line (perfect predictions)
    min_val = min(min(actual), min(predicted))
    max_val = max(max(actual), max(predicted))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    # Add annotations if driver names are provided
    if driver_names is not None:
        for i, name in enumerate(driver_names):
            ax.annotate(name, (actual[i], predicted[i]), 
                       xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Actual Position')
    ax.set_ylabel('Predicted Position')
    ax.set_title('Actual vs Predicted Finishing Positions')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Invert both axes (lower is better in F1)
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    return fig

def main():
    """Main function for the Streamlit app"""
    st.title("🏎️ F1 Race Outcome Prediction")
    
    # Sidebar for model loading and options
    with st.sidebar:
        st.header("Model Settings")
        model_path = st.text_input("Model Path", value="f1_prediction_model.joblib")
        
        if st.button("Load Model"):
            with st.spinner("Loading model..."):
                try:
                    model = load_model(model_path)
                    st.session_state['model'] = model
                    st.success("Model loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {e}")
        
        st.markdown("---")
        st.header("Options")
        show_example = st.checkbox("Show Example Data", value=False)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["Data Upload", "Predictions", "Visualization"])
    
    with tab1:
        st.header("Upload Race Data")
        
        uploaded_file = st.file_uploader("Upload CSV file with race data", type=["csv"])
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state['data'] = data
                st.success(f"Data loaded successfully! ({len(data)} rows, {len(data.columns)} columns)")
                
                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(data.head())
                
                # Check for required columns
                required_columns = ['Position_x', 'best_qualifying_time', 'GridPosition']
                missing_columns = [col for col in required_columns if col not in data.columns]
                
                if missing_columns:
                    st.warning(f"Warning: The following required columns are missing: {', '.join(missing_columns)}")
                    st.info("You can still proceed, but preprocessing will attempt to derive these columns if possible.")
            
            except Exception as e:
                st.error(f"Error loading data: {e}")
        
        elif show_example:
            # Create example data
            st.subheader("Example Data")
            example_data = {
                'Driver': ['VER', 'HAM', 'LEC', 'PER', 'SAI'],
                'Team': ['Red Bull Racing', 'Mercedes', 'Ferrari', 'Red Bull Racing', 'Ferrari'],
                'Position_x': [1, 2, 3, 4, 5],  # Qualifying positions
                'best_qualifying_time': [92.5, 92.7, 92.9, 93.1, 93.3],  # Qualifying times in seconds
                'GridPosition': [1, 2, 3, 4, 5],  # Grid positions
                'prev_race_position': [1, 3, 2, 4, 5],  # Previous race results
                'prev_qualifying_position': [1, 3, 2, 4, 5],  # Previous qualifying positions
                'Position_prev_standings': [1, 2, 3, 4, 5],  # Previous standings positions
                'Points_prev_standings': [250, 220, 190, 170, 150]  # Previous points
            }
            example_df = pd.DataFrame(example_data)
            st.dataframe(example_df)
            
            if st.button("Use Example Data"):
                st.session_state['data'] = example_df
                st.success("Example data loaded!")
    
    with tab2:
        st.header("Make Predictions")
        
        if 'model' not in st.session_state:
            st.warning("Please load the model first from the sidebar.")
        elif 'data' not in st.session_state:
            st.warning("Please upload data or use example data first.")
        else:
            st.info("Model and data are ready for prediction.")
            
            # Preprocess data
            if st.button("Preprocess Data and Make Predictions"):
                with st.spinner("Processing..."):
                    # Get data and model from session state
                    data = st.session_state['data']
                    model = st.session_state['model']
                    
                    # Preprocess data
                    processed_data = preprocess_data(data)
                    
                    # Make predictions
                    try:
                        predictions = model.predict(processed_data)
                        
                        # Create results dataframe
                        results = pd.DataFrame()
                        
                        # Add driver and team information if available
                        if 'Driver' in data.columns:
                            results['Driver'] = data['Driver']
                        
                        if 'Team' in data.columns:
                            results['Team'] = data['Team']
                        elif 'Team_x' in data.columns:
                            results['Team'] = data['Team_x']
                        
                        # Add original features and predictions
                        for col in ['Position_x', 'GridPosition']:
                            if col in data.columns:
                                results[col] = data[col]
                        
                        results['PredictedPosition'] = predictions
                        
                        # Add actual positions if available
                        if 'Position_y' in data.columns:
                            results['ActualPosition'] = data['Position_y']
                            results['Error'] = data['Position_y'] - predictions
                        
                        # Sort by predicted position
                        results = results.sort_values('PredictedPosition')
                        
                        # Store results in session state
                        st.session_state['results'] = results
                        st.session_state['predictions'] = predictions
                        
                        # Display results
                        st.subheader("Prediction Results")
                        st.dataframe(results)
                        
                        # Option to download results
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv,
                            file_name="f1_predictions.csv",
                            mime="text/csv"
                        )
                    
                    except Exception as e:
                        st.error(f"Error making predictions: {e}")
    
    with tab3:
        st.header("Visualization")
        
        if 'results' not in st.session_state:
            st.warning("Please make predictions first.")
        else:
            results = st.session_state['results']
            
            # Check if we have actual positions for error analysis
            has_actual = 'ActualPosition' in results.columns
            
            if has_actual:
                # Plot prediction errors
                st.subheader("Prediction Error Distribution")
                error_fig = plot_prediction_errors(
                    results['ActualPosition'], 
                    results['PredictedPosition']
                )
                st.pyplot(error_fig)
                
                # Plot actual vs predicted
                st.subheader("Actual vs Predicted Positions")
                driver_names = results['Driver'] if 'Driver' in results.columns else None
                comparison_fig = plot_actual_vs_predicted(
                    results['ActualPosition'], 
                    results['PredictedPosition'],
                    driver_names
                )
                st.pyplot(comparison_fig)
                
                # Calculate metrics
                mae = np.mean(np.abs(results['Error']))
                rmse = np.sqrt(np.mean(np.square(results['Error'])))
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Absolute Error", f"{mae:.2f}")
                with col2:
                    st.metric("Root Mean Squared Error", f"{rmse:.2f}")
            
            else:
                st.info("Actual positions are not available in the data. Cannot calculate prediction errors.")
                
                # Just show the predicted positions
                st.subheader("Predicted Finishing Order")
                
                # Create a bar chart of predicted positions
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Get driver names or indices
                drivers = results['Driver'] if 'Driver' in results.columns else results.index
                
                # Create bar chart
                bars = ax.bar(drivers, results['PredictedPosition'], alpha=0.7)
                
                # Customize chart
                ax.set_xlabel('Driver')
                ax.set_ylabel('Predicted Position')
                ax.set_title('Predicted Finishing Positions')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Invert y-axis (lower is better in F1)
                ax.invert_yaxis()
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom')
                
                # Display the chart
                st.pyplot(fig)

if __name__ == "__main__":
    main()