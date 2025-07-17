"""
F1 Race Outcome Prediction - Feature Engineering Suggestions
===========================================================

This file contains three feature engineering ideas that could improve the model's R² score
beyond the current ~0.08 value. Each suggestion includes a code snippet demonstrating
how to implement it.

These suggestions can be incorporated into the F1PredictionModel class by modifying
the prepare_features method.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans

def suggestion_1_weather_track_conditions():
    """
    Suggestion 1: Incorporate Weather and Track Conditions
    ------------------------------------------------------
    
    Weather and track conditions significantly impact race performance. Wet tracks,
    temperature, and humidity affect tire grip, strategy, and overall performance.
    
    Implementation:
    1. Collect weather data for each race (temperature, precipitation, humidity)
    2. Create interaction features between weather and driver/team performance
    3. Add track condition ratings (dry, damp, wet)
    
    Expected impact: Weather-sensitive drivers/teams show distinct performance patterns
    in different conditions. This could significantly improve prediction accuracy.
    """
    # Example code snippet
    def add_weather_features(df, weather_data):
        """
        Add weather-related features to the dataset
        
        Parameters:
        -----------
        df : DataFrame
            The main feature dataframe
        weather_data : DataFrame
            DataFrame containing weather information for each race
            
        Returns:
        --------
        DataFrame
            Enhanced dataframe with weather features
        """
        # Merge weather data with main dataframe
        df = pd.merge(df, weather_data, on=['GP', 'Round'], how='left')
        
        # Create interaction features
        df['qualifying_in_wet'] = (df['best_qualifying_time'] * 
                                  df['precipitation_qualifying'].apply(lambda x: 1.5 if x > 0 else 1.0))
        
        # Create track condition categories
        conditions = ['dry', 'damp', 'wet']
        df['track_condition'] = pd.cut(df['precipitation_race'], 
                                      bins=[-0.1, 0.5, 5, 100], 
                                      labels=conditions)
        
        # One-hot encode track conditions
        track_condition_dummies = pd.get_dummies(df['track_condition'], prefix='track')
        df = pd.concat([df, track_condition_dummies], axis=1)
        
        # Create temperature-related features
        df['temp_variation'] = df['temp_max'] - df['temp_min']
        df['optimal_temp_range'] = ((df['temp_max'] >= 20) & (df['temp_max'] <= 30)).astype(int)
        
        return df
    
    # Example of how to use this in the model pipeline
    print("""
    # In F1PredictionModel.prepare_features:
    
    # Load weather data
    weather_data = pd.read_csv('weather_data.csv')
    
    # Add weather features
    df = add_weather_features(df, weather_data)
    
    # Add new weather features to feature_cols
    weather_features = ['qualifying_in_wet', 'track_dry', 'track_damp', 
                        'track_wet', 'temp_variation', 'optimal_temp_range']
    feature_cols.extend(weather_features)
    """)
    
    return "Weather and track conditions features could improve R² by capturing performance variations in different conditions"

def suggestion_2_driver_team_historical_performance():
    """
    Suggestion 2: Driver-Team Historical Performance Metrics
    -------------------------------------------------------
    
    Create more sophisticated historical performance metrics for driver-team combinations.
    Current and historical performance trends provide valuable context beyond simple
    previous race positions.
    
    Implementation:
    1. Calculate rolling averages of performance metrics
    2. Create driver-team specific features
    3. Calculate performance trends and momentum indicators
    
    Expected impact: These features capture longer-term performance patterns and
    recent momentum, which are strong predictors of race outcomes.
    """
    # Example code snippet
    def add_historical_performance_features(df):
        """
        Add historical performance features to the dataset
        
        Parameters:
        -----------
        df : DataFrame
            The main feature dataframe
            
        Returns:
        --------
        DataFrame
            Enhanced dataframe with historical performance features
        """
        # Ensure data is sorted by driver and round
        df = df.sort_values(['Driver ID', 'Round'])
        
        # Calculate rolling averages (last 3 races)
        df['position_avg_3races'] = df.groupby('Driver ID')['Position_y'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        # Calculate position improvement/decline trend
        df['position_trend'] = df.groupby('Driver ID')['Position_y'].transform(
            lambda x: x.rolling(window=3, min_periods=2).apply(
                lambda y: -1 * np.polyfit(range(len(y)), y, 1)[0] if len(y) >= 2 else 0
            )
        )
        
        # Create driver-team combination features
        df['driver_team_id'] = df['Driver ID'].astype(str) + '_' + df['Team_x'].astype(str)
        
        # Calculate team performance at specific tracks
        df['team_track_performance'] = df.groupby(['Team_x', 'GP'])['Position_y'].transform('mean')
        
        # Calculate driver performance relative to teammate
        teammate_perf = df.groupby(['Team_x', 'Round']).apply(
            lambda x: x.set_index('Driver ID')['Position_y']
        ).reset_index()
        
        # Pivot to get teammate's position
        teammate_perf = teammate_perf.pivot_table(
            index=['Team_x', 'Round'], 
            columns='Driver ID', 
            values='Position_y'
        ).reset_index()
        
        # Melt to get teammate's position for each driver
        teammate_cols = [col for col in teammate_perf.columns if col not in ['Team_x', 'Round']]
        teammate_perf = pd.melt(
            teammate_perf, 
            id_vars=['Team_x', 'Round'], 
            value_vars=teammate_cols,
            var_name='Driver ID', 
            value_name='teammate_position'
        )
        
        # Merge teammate performance back to main dataframe
        df = pd.merge(df, teammate_perf, on=['Team_x', 'Round', 'Driver ID'], how='left')
        
        # Calculate performance vs teammate
        df['vs_teammate'] = df['Position_y'] - df['teammate_position']
        
        return df
    
    # Example of how to use this in the model pipeline
    print("""
    # In F1PredictionModel.prepare_features:
    
    # Add historical performance features
    df = add_historical_performance_features(df)
    
    # Add new historical features to feature_cols
    historical_features = ['position_avg_3races', 'position_trend', 
                          'team_track_performance', 'vs_teammate']
    feature_cols.extend(historical_features)
    """)
    
    return "Historical performance metrics could improve R² by capturing longer-term patterns and team dynamics"

def suggestion_3_advanced_feature_transformations():
    """
    Suggestion 3: Advanced Feature Transformations and Interactions
    --------------------------------------------------------------
    
    Apply advanced feature transformations and create interaction features to capture
    non-linear relationships and complex patterns in the data.
    
    Implementation:
    1. Create polynomial features for key numerical variables
    2. Apply clustering to identify performance groups
    3. Create interaction terms between important features
    
    Expected impact: These transformations can capture complex non-linear relationships
    that simple linear models might miss, potentially improving R² significantly.
    """
    # Example code snippet
    def add_advanced_transformations(X):
        """
        Add advanced feature transformations to the dataset
        
        Parameters:
        -----------
        X : DataFrame
            The feature dataframe
            
        Returns:
        --------
        DataFrame
            Enhanced dataframe with transformed features
        """
        # Create a copy of the input data
        X_enhanced = X.copy()
        
        # Select numerical features for polynomial transformation
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col not in ['Team_x', 'Team']]
        
        # Create polynomial features (degree 2)
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        poly_features = poly.fit_transform(X[numerical_cols])
        
        # Create feature names for polynomial features
        poly_feature_names = []
        for i, feat1 in enumerate(numerical_cols):
            poly_feature_names.append(feat1)  # Original feature
            for j in range(i, len(numerical_cols)):
                feat2 = numerical_cols[j]
                poly_feature_names.append(f"{feat1}_{feat2}")  # Interaction or squared term
        
        # Add polynomial features to the dataframe
        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names[:poly_features.shape[1]])
        
        # Select only the most important polynomial features to avoid overfitting
        important_poly_features = [col for col in poly_df.columns if 
                                  ('Position_x' in col or 'GridPosition' in col or 'best_qualifying_time' in col)]
        
        X_enhanced = pd.concat([X_enhanced, poly_df[important_poly_features]], axis=1)
        
        # Apply clustering to identify performance groups
        if len(X) > 10:  # Only if we have enough samples
            kmeans = KMeans(n_clusters=3, random_state=42)
            X_enhanced['performance_cluster'] = kmeans.fit_predict(X[numerical_cols])
            
            # One-hot encode the clusters
            cluster_dummies = pd.get_dummies(X_enhanced['performance_cluster'], 
                                           prefix='cluster')
            X_enhanced = pd.concat([X_enhanced, cluster_dummies], axis=1)
        
        # Create ratio features
        if 'best_qualifying_time' in X.columns and 'Position_x' in X.columns:
            X_enhanced['quali_time_per_position'] = X['best_qualifying_time'] / X['Position_x']
        
        if 'GridPosition' in X.columns and 'Position_x' in X.columns:
            X_enhanced['grid_quali_ratio'] = X['GridPosition'] / X['Position_x']
        
        return X_enhanced
    
    # Example of how to use this in the model pipeline
    print("""
    # In F1PredictionModel.prepare_features:
    
    # Apply advanced transformations to features
    X = add_advanced_transformations(X)
    
    # Note: When using this approach, you may need to adjust the model parameters
    # to prevent overfitting, such as increasing regularization or reducing max_depth
    """)
    
    return "Advanced feature transformations could improve R² by capturing non-linear relationships and complex patterns"

def main():
    """Display all feature engineering suggestions"""
    print("F1 Race Outcome Prediction - Feature Engineering Suggestions")
    print("===========================================================\n")
    
    print("SUGGESTION 1: WEATHER AND TRACK CONDITIONS")
    print("------------------------------------------")
    result1 = suggestion_1_weather_track_conditions()
    print(f"Expected impact: {result1}\n")
    
    print("SUGGESTION 2: DRIVER-TEAM HISTORICAL PERFORMANCE METRICS")
    print("-------------------------------------------------------")
    result2 = suggestion_2_driver_team_historical_performance()
    print(f"Expected impact: {result2}\n")
    
    print("SUGGESTION 3: ADVANCED FEATURE TRANSFORMATIONS AND INTERACTIONS")
    print("--------------------------------------------------------------")
    result3 = suggestion_3_advanced_feature_transformations()
    print(f"Expected impact: {result3}\n")
    
    print("Implementation:")
    print("To implement these suggestions, modify the prepare_features method in the")
    print("F1PredictionModel class to incorporate the desired feature engineering techniques.")
    print("You may need to collect additional data (like weather information) for some suggestions.")

if __name__ == "__main__":
    main()