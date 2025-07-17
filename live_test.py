"""
Live Race Testing Script
=======================

Test your model with real qualifying data to predict race outcomes.
This script simulates using the model for an upcoming race.
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

def load_real_qualifying_data():
    """Load real qualifying data from one of your GP folders."""
    
    # Use Austria GP as example
    quali_file = "Austria GP/qualifying_results_2025_round_11.csv"
    
    try:
        df = pd.read_csv(quali_file)
        print(f"✓ Loaded qualifying data: {len(df)} drivers")
        return df
    except FileNotFoundError:
        print(f"❌ Could not find {quali_file}")
        return None

def prepare_qualifying_for_prediction(quali_df):
    """Convert qualifying data to model input format."""
    
    # Convert time strings to seconds
    def time_to_seconds(time_str):
        if pd.isna(time_str) or time_str == '':
            return np.nan
        try:
            # Handle pandas timedelta format
            if 'days' in str(time_str):
                return pd.to_timedelta(time_str).total_seconds()
            else:
                return float(time_str)
        except:
            return np.nan
    
    # Process qualifying times
    for col in ['Q1 Time', 'Q2 Time', 'Q3 Time']:
        if col in quali_df.columns:
            quali_df[col + '_seconds'] = quali_df[col].apply(time_to_seconds)
    
    # Calculate best qualifying time
    quali_df['best_qualifying_time'] = quali_df.apply(
        lambda row: row.get('Q3 Time_seconds', np.nan) if pd.notna(row.get('Q3 Time_seconds', np.nan))
               else (row.get('Q2 Time_seconds', np.nan) if pd.notna(row.get('Q2 Time_seconds', np.nan))
                     else row.get('Q1 Time_seconds', np.nan)),
        axis=1
    )
    
    # Prepare model input features
    model_input = pd.DataFrame({
        'Position_x': quali_df['Position'].values,
        'best_qualifying_time': quali_df['best_qualifying_time'].values,
        'GridPosition': quali_df['Position'].values,  # Assuming grid = quali position
        'grid_position_diff': [0] * len(quali_df),  # No data for this
        'Team': quali_df['Team'].values
    })
    
    # Fill NaN values with median
    model_input['best_qualifying_time'].fillna(
        model_input['best_qualifying_time'].median(), inplace=True
    )
    
    return model_input, quali_df

def predict_race_outcome():
    """Main function to predict race outcome from qualifying."""
    
    print("🏎️ F1 Live Race Prediction")
    print("=" * 40)
    
    # Load qualifying data
    quali_df = load_real_qualifying_data()
    if quali_df is None:
        return
    
    print(f"📊 Qualifying Results Preview:")
    print(quali_df[['Full Name', 'Team', 'Position']].head(10).to_string(index=False))
    
    # Prepare data for model
    model_input, original_quali = prepare_qualifying_for_prediction(quali_df)
    
    # Load model
    try:
        model = joblib.load('f1_prediction_model.joblib')
        print(f"\n✓ Model loaded successfully")
    except FileNotFoundError:
        print(f"❌ No trained model found. Run 'python run_f1_model.py' first.")
        return
    
    # Make predictions
    try:
        predictions = model.predict(model_input)
        
        # Create results dataframe
        results = pd.DataFrame({
            'Driver': original_quali['Full Name'],
            'Team': original_quali['Team'],
            'Quali_Pos': original_quali['Position'],
            'Predicted_Finish': predictions.round(1),
            'Expected_Change': (predictions - original_quali['Position']).round(1)
        })
        
        # Sort by predicted finishing position
        results = results.sort_values('Predicted_Finish')
        
        print(f"\n🏁 RACE PREDICTION RESULTS")
        print("=" * 40)
        print(results.to_string(index=False))
        
        # Analysis
        print(f"\n📈 Race Analysis:")
        
        # Biggest movers
        biggest_gainer = results.loc[results['Expected_Change'].idxmin()]
        biggest_loser = results.loc[results['Expected_Change'].idxmax()]
        
        print(f"  🚀 Biggest Expected Gainer: {biggest_gainer['Driver']} ({biggest_gainer['Expected_Change']:+.1f} positions)")
        print(f"  📉 Biggest Expected Loser: {biggest_loser['Driver']} ({biggest_loser['Expected_Change']:+.1f} positions)")
        
        # Podium prediction
        podium = results.head(3)
        print(f"\n🏆 Predicted Podium:")
        for i, (_, driver) in enumerate(podium.iterrows(), 1):
            medal = ["🥇", "🥈", "🥉"][i-1]
            print(f"  {medal} P{i}: {driver['Driver']} ({driver['Team']})")
        
        # Points predictions (F1 points system: 25,18,15,12,10,8,6,4,2,1)
        points_system = [25,18,15,12,10,8,6,4,2,1] + [0]*10
        results['Predicted_Points'] = [points_system[int(pos)-1] if pos <= 10 else 0 
                                     for pos in results['Predicted_Finish']]
        
        print(f"\n🏁 Predicted Points Scorers:")
        points_scorers = results[results['Predicted_Points'] > 0]
        for _, driver in points_scorers.iterrows():
            print(f"  P{driver['Predicted_Finish']:.0f}: {driver['Driver']:<20} {driver['Predicted_Points']:.0f} pts")
        
        # Save predictions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"race_prediction_{timestamp}.csv"
        results.to_csv(filename, index=False)
        print(f"\n💾 Predictions saved to: {filename}")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        print("This might be due to feature mismatch or data issues")

def compare_with_actual():
    """Compare predictions with actual race results."""
    
    print(f"\n🔍 Comparing with Actual Results")
    print("=" * 40)
    
    # Load actual race results
    try:
        race_results = pd.read_csv("Austria GP/race_results_2025_round_11.csv")
        print(f"✓ Loaded actual race results")
        
        # Load latest prediction
        import glob
        pred_files = glob.glob("race_prediction_*.csv")
        if pred_files:
            latest_pred = max(pred_files)
            predictions = pd.read_csv(latest_pred)
            
            # Merge predictions with actual results
            comparison = pd.merge(
                predictions[['Driver', 'Predicted_Finish']], 
                race_results[['FullName', 'Position']], 
                left_on='Driver', right_on='FullName', 
                how='inner'
            )
            
            # Calculate accuracy
            comparison['Difference'] = comparison['Position'] - comparison['Predicted_Finish']
            comparison['Abs_Difference'] = abs(comparison['Difference'])
            
            print(f"\n📊 Prediction vs Reality:")
            display_cols = ['Driver', 'Predicted_Finish', 'Position', 'Difference']
            print(comparison[display_cols].to_string(index=False))
            
            # Accuracy metrics
            mae = comparison['Abs_Difference'].mean()
            perfect_predictions = sum(comparison['Abs_Difference'] == 0)
            within_2 = sum(comparison['Abs_Difference'] <= 2)
            
            print(f"\n📈 Accuracy Summary:")
            print(f"  • Mean Absolute Error: {mae:.2f} positions")
            print(f"  • Perfect predictions: {perfect_predictions}/{len(comparison)}")
            print(f"  • Within ±2 positions: {within_2}/{len(comparison)} ({within_2/len(comparison)*100:.1f}%)")
            
        else:
            print("❌ No prediction files found")
            
    except FileNotFoundError:
        print("❌ Actual race results not found")

if __name__ == "__main__":
    predict_race_outcome()
    compare_with_actual()
