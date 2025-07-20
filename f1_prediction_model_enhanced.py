"""
Enhanced F1 Prediction Model with Advanced Feature Engineering
==============================================================

This enhanced version of the F1 prediction model implements:
1. Historical performance metrics and rolling averages
2. Advanced feature transformations and polynomial interactions  
3. Team and driver-specific performance patterns
4. Improved model selection and ensemble methods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import os
import glob
import joblib
import warnings
warnings.filterwarnings('ignore')

class F1PredictionModelEnhanced:
    """
    Enhanced F1 prediction model with advanced feature engineering and ensemble methods.
    """

    def __init__(self, data_dir=".", model_type="ensemble", use_polynomial=True, max_features=None):
        """
        Initialize the enhanced F1 prediction model.

        Parameters:
        -----------
        data_dir : str
            Directory containing the F1 data files
        model_type : str
            Type of model to use ('random_forest', 'gradient_boosting', 'ensemble')
        use_polynomial : bool
            Whether to use polynomial features
        max_features : int
            Maximum number of features to select (None for all)
        """
        self.data_dir = data_dir
        self.model_type = model_type
        self.use_polynomial = use_polynomial
        self.max_features = max_features
        self.model = None
        self.preprocessor = None
        self.feature_importances = None
        self.feature_names = None

    def load_data(self, gp_folders=None):
        """Load and combine data from multiple Grand Prix events."""
        if gp_folders is None:
            # Find all GP folders
            gp_folders = [d for d in os.listdir(self.data_dir) 
                         if os.path.isdir(os.path.join(self.data_dir, d)) and "GP" in d]

        # Initialize empty DataFrames
        qualifying_results = pd.DataFrame()
        race_results = pd.DataFrame()
        driver_standings = pd.DataFrame()
        lap_times = pd.DataFrame()

        # Load data from each GP folder
        for gp_folder in gp_folders:
            gp_path = os.path.join(self.data_dir, gp_folder)

            # Extract round number from folder name or files
            round_num = None
            if "round_" in gp_folder.lower():
                try:
                    round_num = int(gp_folder.lower().split("round_")[1])
                except:
                    pass
            
            # Try to extract from file names if not found in folder name
            if round_num is None:
                csv_files = glob.glob(os.path.join(gp_path, "*round_*.csv"))
                if csv_files:
                    try:
                        round_num = int(csv_files[0].split("round_")[1].split(".")[0])
                    except:
                        pass

            # Load qualifying results
            qual_file = glob.glob(os.path.join(gp_path, "qualifying_results_*.csv"))
            if qual_file:
                df = pd.read_csv(qual_file[0])
                if round_num:
                    df['Round'] = round_num
                df['GP'] = gp_folder
                qualifying_results = pd.concat([qualifying_results, df])

            # Load race results
            race_file = glob.glob(os.path.join(gp_path, "race_results_*.csv"))
            if race_file:
                df = pd.read_csv(race_file[0])
                if round_num:
                    df['Round'] = round_num
                df['GP'] = gp_folder
                race_results = pd.concat([race_results, df])

            # Load driver standings
            standings_file = glob.glob(os.path.join(gp_path, "driver_standings_*.csv"))
            if standings_file:
                df = pd.read_csv(standings_file[0])
                if round_num:
                    df['Round'] = round_num
                df['GP'] = gp_folder
                driver_standings = pd.concat([driver_standings, df])

            # Load lap times for additional features
            lap_file = glob.glob(os.path.join(gp_path, "lap_times_*.csv"))
            if lap_file:
                df = pd.read_csv(lap_file[0])
                if round_num:
                    df['Round'] = round_num
                df['GP'] = gp_folder
                lap_times = pd.concat([lap_times, df])

        return {
            'qualifying_results': qualifying_results,
            'race_results': race_results,
            'driver_standings': driver_standings,
            'lap_times': lap_times
        }

    def add_historical_performance_features(self, df):
        """Add historical performance features based on previous races."""
        # Check if Round column exists, if not create a simple sequence
        if 'Round' not in df.columns:
            # Create round numbers based on GP order
            gp_order = {'Canada GP': 10, 'Austria GP': 11, 'UK GP': 12}
            df['Round'] = df['GP'].map(gp_order).fillna(df.groupby('GP').ngroup() + 1)
        
        # Ensure data is sorted by driver and round
        df = df.sort_values(['Driver ID', 'Round'])
        
        # Calculate rolling averages for last 3 races (shift by 1 to avoid data leakage)
        df['position_avg_3races'] = df.groupby('Driver ID')['Position_y'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
        )
        
        # Calculate qualifying position rolling average
        df['qualifying_avg_3races'] = df.groupby('Driver ID')['Position_x'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
        )
        
        # Calculate position improvement/decline trend
        df['position_trend'] = df.groupby('Driver ID')['Position_y'].transform(
            lambda x: x.rolling(window=3, min_periods=2).apply(
                lambda y: -1 * np.polyfit(range(len(y)), y, 1)[0] if len(y) >= 2 else 0
            ).shift(1)
        )
        
        # Calculate consistency (standard deviation of positions)
        df['position_consistency'] = df.groupby('Driver ID')['Position_y'].transform(
            lambda x: x.rolling(window=3, min_periods=2).std().shift(1)
        )
        
        # Calculate team performance at specific tracks (mean team position for this track)
        df['team_track_performance'] = df.groupby(['Team_x', 'GP'])['Position_y'].transform('mean')
        
        # Calculate driver performance relative to teammate
        df['teammate_comparison'] = df.groupby(['Team_x', 'Round']).apply(
            lambda x: self._calculate_teammate_comparison(x)
        ).reset_index(level=[0,1], drop=True)
        
        return df

    def _calculate_teammate_comparison(self, team_round_data):
        """Calculate performance relative to teammate."""
        if len(team_round_data) == 2:
            positions = team_round_data['Position_y'].values
            return pd.Series(positions - positions.mean(), index=team_round_data.index)
        else:
            return pd.Series([0] * len(team_round_data), index=team_round_data.index)

    def add_lap_time_features(self, df, lap_times):
        """Add lap time based features."""
        if lap_times.empty:
            return df
            
        # Calculate average lap time for each driver (only if lap times are available)
        try:
            avg_lap_times = lap_times.groupby(['Driver ID', 'GP', 'Round'])['Time'].apply(
                lambda x: pd.to_numeric(x.str.replace(':', '').str.replace('.', ''), errors='coerce').mean()
            ).reset_index()
            avg_lap_times.columns = ['Driver ID', 'GP', 'Round', 'avg_lap_time']
            
            # Calculate fastest lap time
            fastest_lap_times = lap_times.groupby(['Driver ID', 'GP', 'Round'])['Time'].apply(
                lambda x: pd.to_numeric(x.str.replace(':', '').str.replace('.', ''), errors='coerce').min()
            ).reset_index()
            fastest_lap_times.columns = ['Driver ID', 'GP', 'Round', 'fastest_lap_time']
            
            # Merge with main dataframe
            df = pd.merge(df, avg_lap_times, on=['Driver ID', 'GP', 'Round'], how='left')
            df = pd.merge(df, fastest_lap_times, on=['Driver ID', 'GP', 'Round'], how='left')
        except Exception as e:
            print(f"Warning: Could not process lap times: {e}")
        
        return df

    def add_advanced_features(self, df):
        """Add advanced calculated features."""
        # Qualifying time relative to pole position
        df['quali_time_vs_pole'] = df.groupby('GP')['best_qualifying_time'].transform(
            lambda x: x - x.min()
        )
        
        # Position improvement from qualifying to grid
        df['quali_to_grid_change'] = df['GridPosition'] - df['Position_x']
        
        # Team performance score (based on constructor standings)
        team_scores = df.groupby('Team_x')['Position_y'].mean()
        df['team_performance_score'] = df['Team_x'].map(team_scores)
        
        # Create driver experience proxy (count of races)
        df['driver_experience'] = df.groupby('Driver ID').cumcount()
        
        # Grid position normalized by field size
        df['grid_position_normalized'] = df['GridPosition'] / df.groupby('GP')['GridPosition'].transform('max')
        
        return df

    def prepare_features(self, data):
        """
        Prepare enhanced features for the model.
        """
        # Extract DataFrames from the data dictionary
        qualifying_results = data['qualifying_results']
        race_results = data['race_results']
        driver_standings = data['driver_standings']
        lap_times = data['lap_times']

        # Merge qualifying and race results
        df = pd.merge(qualifying_results, race_results, 
                      left_on=['Driver ID'], 
                      right_on=['DriverId'], 
                      how='inner',
                      suffixes=('_qual', '_race'))

        # Clean up column names for consistency
        df['Position_x'] = df['Position_qual']  # Qualifying position
        df['Position_y'] = df['Position_race']  # Race position  
        df['Team_x'] = df['Team']  # Team from qualifying data
        
        # Ensure GP column exists by taking it from qualifying data
        if 'GP' not in df.columns:
            # Add GP from qualifying_results if it exists there
            if 'GP' in qualifying_results.columns:
                gp_mapping = qualifying_results.groupby('Driver ID')['GP'].first().to_dict()
                df['GP'] = df['Driver ID'].map(gp_mapping)
            else:
                df['GP'] = 'Unknown GP'

        # Convert time strings to seconds
        for col in ['Q1 Time', 'Q2 Time', 'Q3 Time']:
            if col in df.columns:
                df[col + '_seconds'] = df[col].apply(
                    lambda x: pd.to_timedelta(x).total_seconds() if pd.notna(x) else np.nan
                )

        # Calculate best qualifying time
        df['best_qualifying_time'] = df.apply(
            lambda row: row['Q3 Time_seconds'] if pd.notna(row['Q3 Time_seconds']) 
                   else (row['Q2 Time_seconds'] if pd.notna(row['Q2 Time_seconds']) 
                         else row['Q1 Time_seconds']),
            axis=1
        )

        # Calculate grid position difference
        df['grid_position_diff'] = df['GridPosition'] - df['Position_y']

        # Add historical performance features
        df = self.add_historical_performance_features(df)
        
        # Add lap time features
        df = self.add_lap_time_features(df, lap_times)
        
        # Add advanced features
        df = self.add_advanced_features(df)

        # Merge with driver standings from previous round
        if 'Round' in df.columns and not driver_standings.empty:
            try:
                prev_standings = driver_standings.copy()
                prev_standings['Round'] = prev_standings['Round'] + 1
                df = pd.merge(df, prev_standings[['Driver ID', 'Round', 'Points', 'Position']], 
                              on=['Driver ID', 'Round'], 
                              how='left', 
                              suffixes=('', '_prev_standings'))
            except Exception as e:
                print(f"Warning: Could not merge driver standings: {e}")

        # Select enhanced features
        feature_cols = [
            # Basic features
            'Position_x',  # Qualifying position
            'best_qualifying_time',
            'GridPosition',
            'grid_position_diff',
            
            # Historical performance features
            'position_avg_3races',
            'qualifying_avg_3races', 
            'position_trend',
            'position_consistency',
            'team_track_performance',
            'teammate_comparison',
            
            # Advanced features
            'quali_time_vs_pole',
            'quali_to_grid_change',
            'team_performance_score',
            'driver_experience',
            'grid_position_normalized',
        ]

        # Add lap time features if available
        if 'avg_lap_time' in df.columns:
            feature_cols.extend(['avg_lap_time', 'fastest_lap_time'])

        # Add previous standings features if available
        if 'Position_prev_standings' in df.columns:
            feature_cols.extend(['Position_prev_standings', 'Points_prev_standings'])

        # Add team as categorical feature
        if 'Team_x' in df.columns:
            feature_cols.append('Team_x')
        elif 'Team' in df.columns:
            feature_cols.append('Team')

        # Prepare X and y, keeping only features that exist
        existing_features = [col for col in feature_cols if col in df.columns]
        X = df[existing_features].copy()
        y = df['Position_y'].copy()

        # Handle missing values
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')

        self.feature_names = list(X.columns)
        
        return X, y, df

    def create_polynomial_features(self, X):
        """Create polynomial and interaction features."""
        if not self.use_polynomial:
            return X
            
        # Select numerical features for polynomial transformation
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        if len(numerical_cols) == 0:
            return X
            
        # Create polynomial features (degree 2, interactions only to avoid explosion)
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        poly_features = poly.fit_transform(X[numerical_cols])
        
        # Get feature names
        poly_feature_names = poly.get_feature_names_out(numerical_cols)
        
        # Create polynomial DataFrame
        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=X.index)
        
        # Keep only interaction terms (not the original features and squares)
        interaction_cols = [col for col in poly_df.columns if ' ' in col]
        poly_df = poly_df[interaction_cols]
        
        # Combine with categorical features
        result_df = pd.concat([X[categorical_cols], X[numerical_cols], poly_df], axis=1)
        
        return result_df

    def build_model(self, X):
        """Build enhanced machine learning pipeline."""
        # Define categorical and numerical features
        categorical_features = [col for col in X.columns if X[col].dtype == 'object']
        numerical_features = [col for col in X.columns if col not in categorical_features]

        # Define preprocessing for numerical features
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        # Define preprocessing for categorical features
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Combine preprocessing steps
        transformers = [('num', numerical_transformer, numerical_features)]
        if categorical_features:
            transformers.append(('cat', categorical_transformer, categorical_features))

        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

        # Feature selection if max_features is specified
        feature_selector = None
        if self.max_features and self.max_features < len(X.columns):
            feature_selector = SelectKBest(score_func=f_regression, k=self.max_features)

        # Choose model based on model_type
        if self.model_type == 'ensemble':
            # Create ensemble of different models
            rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
            gb = GradientBoostingRegressor(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42)
            ridge = Ridge(alpha=1.0)
            
            model = VotingRegressor([
                ('rf', rf),
                ('gb', gb), 
                ('ridge', ridge)
            ])
            
            param_grid = {
                'model__rf__n_estimators': [150, 200],
                'model__rf__max_depth': [10, 15],
                'model__gb__n_estimators': [100, 150],
                'model__gb__learning_rate': [0.05, 0.1]
            }
        elif self.model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(random_state=42)
            param_grid = {
                'model__n_estimators': [100, 200, 300],
                'model__learning_rate': [0.05, 0.1, 0.15],
                'model__max_depth': [3, 5, 7]
            }
        else:  # random_forest
            model = RandomForestRegressor(random_state=42)
            param_grid = {
                'model__n_estimators': [150, 200, 300],
                'model__max_depth': [10, 15, 20],
                'model__min_samples_split': [2, 5, 10]
            }

        # Create pipeline steps
        pipeline_steps = [('preprocessor', preprocessor)]
        
        if feature_selector:
            pipeline_steps.append(('feature_selector', feature_selector))
            
        pipeline_steps.append(('model', model))

        # Create pipeline
        pipeline = Pipeline(steps=pipeline_steps)

        # Create grid search with better cross-validation
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=3,  # Reduced due to small dataset
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

        return grid_search

    def train(self, X=None, y=None):
        """Train the enhanced model."""
        if X is not None and y is not None:
            self.X = X
            self.y = y

        # Create polynomial features if enabled
        X_enhanced = self.create_polynomial_features(self.X)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_enhanced, self.y, test_size=0.2, random_state=42
        )

        print(f"Training with {self.X_train.shape[1]} features on {self.X_train.shape[0]} samples...")

        # Build and train the model
        self.model = self.build_model(X_enhanced)
        self.model.fit(self.X_train, self.y_train)

        # Get the best model
        self.best_model = self.model.best_estimator_
        
        print(f"Best parameters: {self.model.best_params_}")
        print(f"Best CV score: {-self.model.best_score_:.4f}")

        return self

    def evaluate(self):
        """Evaluate the enhanced model."""
        # Make predictions
        y_pred = self.best_model.predict(self.X_test)

        # Calculate metrics
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        # Print evaluation results
        print(f"\nEnhanced Model Evaluation Results:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")

        # Cross-validation score
        cv_scores = cross_val_score(self.best_model, self.X_test, self.y_test, 
                                   cv=3, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        print(f"Cross-validation RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")

        # Position accuracy metrics
        within_1 = np.sum(np.abs(y_pred - self.y_test) <= 1)
        within_2 = np.sum(np.abs(y_pred - self.y_test) <= 2)
        within_3 = np.sum(np.abs(y_pred - self.y_test) <= 3)
        total = len(self.y_test)
        
        print(f"\nPosition Accuracy:")
        print(f"Within ±1 position: {within_1}/{total} ({within_1/total*100:.1f}%)")
        print(f"Within ±2 positions: {within_2}/{total} ({within_2/total*100:.1f}%)")
        print(f"Within ±3 positions: {within_3}/{total} ({within_3/total*100:.1f}%)")

        # Create visualization
        plt.figure(figsize=(15, 5))
        
        # Actual vs predicted
        plt.subplot(1, 3, 1)
        plt.scatter(self.y_test, y_pred, alpha=0.7, color='blue')
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--')
        plt.xlabel('Actual Position')
        plt.ylabel('Predicted Position')
        plt.title('Enhanced Model: Actual vs Predicted')
        plt.grid(True, alpha=0.3)

        # Residuals plot
        plt.subplot(1, 3, 2)
        residuals = y_pred - self.y_test
        plt.scatter(y_pred, residuals, alpha=0.7, color='green')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Position')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.grid(True, alpha=0.3)

        # Error distribution
        plt.subplot(1, 3, 3)
        plt.hist(residuals, bins=15, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('enhanced_model_evaluation.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Enhanced model visualization saved to 'enhanced_model_evaluation.png'")

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std(),
            'within_1_pos': within_1/total,
            'within_2_pos': within_2/total,
            'within_3_pos': within_3/total
        }

    def predict(self, X):
        """Make predictions with the enhanced model."""
        if self.best_model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        # Apply polynomial features if used during training
        X_enhanced = self.create_polynomial_features(X)
        
        # Ensure features match training features
        missing_cols = set(self.X_train.columns) - set(X_enhanced.columns)
        extra_cols = set(X_enhanced.columns) - set(self.X_train.columns)
        
        if missing_cols:
            print(f"Warning: Missing features {missing_cols}, filling with median values")
            for col in missing_cols:
                if col in self.X.columns:
                    X_enhanced[col] = self.X[col].median() if self.X[col].dtype in ['float64', 'int64'] else self.X[col].mode()[0]
                else:
                    X_enhanced[col] = 0
        
        if extra_cols:
            print(f"Warning: Extra features {extra_cols}, removing them")
            X_enhanced = X_enhanced.drop(columns=list(extra_cols))
        
        # Reorder columns to match training data
        X_enhanced = X_enhanced[self.X_train.columns]

        return self.best_model.predict(X_enhanced)

    def save_model(self, filename='f1_prediction_model_enhanced.joblib'):
        """Save the enhanced model."""
        if self.best_model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        # Save model and metadata
        model_data = {
            'model': self.best_model,
            'feature_names': self.feature_names,
            'use_polynomial': self.use_polynomial,
            'model_type': self.model_type,
            'training_features': list(self.X_train.columns)
        }
        
        joblib.dump(model_data, filename)
        print(f"Enhanced model saved to {filename}")
        return filename

    def load_model(self, filename='f1_prediction_model_enhanced.joblib'):
        """Load the enhanced model."""
        model_data = joblib.load(filename)
        self.best_model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.use_polynomial = model_data['use_polynomial']
        self.model_type = model_data['model_type']
        print(f"Enhanced model loaded from {filename}")
        return self


def main():
    """Run enhanced model training and evaluation."""
    print("F1 Enhanced Prediction Model")
    print("=" * 50)
    
    # Create enhanced model instance
    model = F1PredictionModelEnhanced(
        data_dir=".",
        model_type="ensemble",
        use_polynomial=True,
        max_features=None  # Use all features
    )
    
    print("Loading data...")
    data = model.load_data()
    
    print("Preparing enhanced features...")
    X, y, df = model.prepare_features(data)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {list(X.columns)}")
    
    # Set data in model
    model.X = X
    model.y = y
    
    print("\nTraining enhanced model...")
    model.train()
    
    print("\nEvaluating enhanced model...")
    metrics = model.evaluate()
    
    print("\nSaving enhanced model...")
    model.save_model()
    
    print("\n" + "=" * 50)
    print("ENHANCED MODEL TRAINING COMPLETE!")
    print("=" * 50)
    print(f"Final R² Score: {metrics['r2']:.4f}")
    print(f"Final RMSE: {metrics['rmse']:.4f}")
    print(f"Position accuracy (±2): {metrics['within_2_pos']*100:.1f}%")


if __name__ == "__main__":
    main()