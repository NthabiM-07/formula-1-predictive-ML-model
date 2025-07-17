import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import glob
import joblib

class F1PredictionModel:
    """
    A machine learning model for predicting Formula 1 race outcomes.

    This model uses historical race data, qualifying results, and driver/team information
    to predict the finishing positions of drivers in future races.
    """

    def __init__(self, data_dir=".", model_type="random_forest"):
        """
        Initialize the F1 prediction model.

        Parameters:
        -----------
        data_dir : str
            Directory containing the F1 data files
        model_type : str
            Type of model to use ('random_forest' or 'gradient_boosting')
        """
        self.data_dir = data_dir
        self.model_type = model_type
        self.model = None
        self.preprocessor = None
        self.feature_importances = None

    def load_data(self, gp_folders=None):
        """
        Load and combine data from multiple Grand Prix events.

        Parameters:
        -----------
        gp_folders : list of str
            List of GP folder names to load data from. If None, load all available data.

        Returns:
        --------
        dict of DataFrames
            Dictionary containing the loaded DataFrames
        """
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

            # Extract round number from folder name
            round_num = None
            if "round_" in gp_folder.lower():
                round_num = int(gp_folder.lower().split("round_")[1])

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

            # Load lap times
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

    def prepare_features(self, data):
        """
        Prepare features for the model from the loaded data.

        Parameters:
        -----------
        data : dict of DataFrames
            Dictionary containing the loaded DataFrames

        Returns:
        --------
        X : DataFrame
            Feature matrix
        y : Series
            Target variable (finishing position)
        """
        # Extract DataFrames from the data dictionary
        qualifying_results = data['qualifying_results']
        race_results = data['race_results']
        driver_standings = data['driver_standings']

        # Merge qualifying and race results
        df = pd.merge(qualifying_results, race_results, 
                      left_on=['Driver ID', 'GP'], 
                      right_on=['DriverId', 'GP'], 
                      how='inner')

        # Convert time strings to seconds
        for col in ['Q1 Time', 'Q2 Time', 'Q3 Time']:
            if col in df.columns:
                # Extract seconds from timedelta strings
                df[col + '_seconds'] = df[col].apply(
                    lambda x: pd.to_timedelta(x).total_seconds() if pd.notna(x) else np.nan
                )

        # Calculate qualifying time (use best available time)
        df['best_qualifying_time'] = df.apply(
            lambda row: row['Q3 Time_seconds'] if pd.notna(row['Q3 Time_seconds']) 
                   else (row['Q2 Time_seconds'] if pd.notna(row['Q2 Time_seconds']) 
                         else row['Q1 Time_seconds']),
            axis=1
        )

        # Calculate grid position vs finishing position difference
        df['grid_position_diff'] = df['GridPosition'] - df['Position_y']

        # Add previous race results and standings if available
        if 'Round' in df.columns:
            # Sort by driver and round
            df = df.sort_values(['Driver ID', 'Round'])

            # Add previous race position
            df['prev_race_position'] = df.groupby('Driver ID')['Position_y'].shift(1)

            # Add previous qualifying position
            df['prev_qualifying_position'] = df.groupby('Driver ID')['Position_x'].shift(1)

            # Merge with driver standings from previous round
            prev_standings = driver_standings.copy()
            prev_standings['Round'] = prev_standings['Round'] + 1  # Shift round for joining
            df = pd.merge(df, prev_standings[['Driver ID', 'Round', 'Points', 'Position']], 
                          on=['Driver ID', 'Round'], 
                          how='left', 
                          suffixes=('', '_prev_standings'))

        # Select and prepare features
        feature_cols = [
            'Position_x',  # Qualifying position
            'best_qualifying_time',
            'GridPosition',
        ]

        # Add optional features if available
        optional_features = [
            'prev_race_position', 
            'prev_qualifying_position',
            'Position_prev_standings',  # Previous standings position
            'Points_prev_standings',    # Previous points
            'grid_position_diff'
        ]

        for feature in optional_features:
            if feature in df.columns:
                feature_cols.append(feature)

        # Add team as a categorical feature
        if 'Team_x' in df.columns:
            feature_cols.append('Team_x')
        elif 'Team' in df.columns:
            feature_cols.append('Team')

        # Prepare X and y
        X = df[feature_cols].copy()
        y = df['Position_y'].copy()  # Finishing position as target

        # Handle missing values
        X = X.fillna(X.mean(numeric_only=True))

        return X, y, df

    def build_model(self):
        """
        Build the machine learning pipeline with preprocessing and model.

        Returns:
        --------
        Pipeline
            Scikit-learn pipeline with preprocessor and model
        """
        # Define categorical and numerical features
        categorical_features = ['Team_x'] if 'Team_x' in self.X.columns else ['Team'] if 'Team' in self.X.columns else []
        numerical_features = [col for col in self.X.columns if col not in categorical_features]

        # Define preprocessing for numerical features
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        # Define preprocessing for categorical features
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features) if categorical_features else None
            ],
            remainder='drop'
        )

        # Remove None transformers
        preprocessor.transformers = [t for t in preprocessor.transformers if t is not None]

        # Choose model based on model_type
        if self.model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(random_state=42)
            param_grid = {
                'model__n_estimators': [100, 200],
                'model__learning_rate': [0.05, 0.1],
                'model__max_depth': [3, 5]
            }
        else:  # default to random_forest
            model = RandomForestRegressor(random_state=42)
            param_grid = {
                'model__n_estimators': [100, 200],
                'model__max_depth': [None, 10, 20],
                'model__min_samples_split': [2, 5]
            }

        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        # Create grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        return grid_search

    def train(self, X=None, y=None):
        """
        Train the model on the provided data.

        Parameters:
        -----------
        X : DataFrame, optional
            Feature matrix. If None, use the data loaded with load_data()
        y : Series, optional
            Target variable. If None, use the data loaded with load_data()

        Returns:
        --------
        self
            Trained model instance
        """
        if X is not None and y is not None:
            self.X = X
            self.y = y

        # Split data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # Build and train the model
        self.model = self.build_model()
        self.model.fit(self.X_train, self.y_train)

        # Get the best model
        self.best_model = self.model.best_estimator_

        # Extract feature importances if available
        if hasattr(self.best_model['model'], 'feature_importances_'):
            # Get feature names after preprocessing
            preprocessor = self.best_model['preprocessor']
            feature_names = []

            # Get numerical feature names
            numerical_features = [col for col in self.X.columns 
                                if col not in ['Team_x', 'Team']]
            feature_names.extend(numerical_features)

            # Get categorical feature names if any
            categorical_features = ['Team_x'] if 'Team_x' in self.X.columns else ['Team'] if 'Team' in self.X.columns else []
            if categorical_features:
                # Get the OneHotEncoder
                cat_idx = [i for i, (name, _, _) in enumerate(preprocessor.transformers_) 
                          if name == 'cat'][0]
                cat_transformer = preprocessor.transformers_[cat_idx][1].named_steps['onehot']

                # Get the categories
                cat_features = []
                for i, col in enumerate(categorical_features):
                    categories = cat_transformer.categories_[i]
                    cat_features.extend([f"{col}_{cat}" for cat in categories])

                feature_names.extend(cat_features)

            # Get feature importances
            importances = self.best_model['model'].feature_importances_

            # Create a DataFrame of feature importances
            self.feature_importances = pd.DataFrame({
                'Feature': feature_names[:len(importances)],
                'Importance': importances
            }).sort_values('Importance', ascending=False)

        return self

    def evaluate(self):
        """
        Evaluate the model on the test set.

        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        # Make predictions on the test set
        y_pred = self.best_model.predict(self.X_test)

        # Calculate metrics
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        # Print evaluation results
        print(f"Model Evaluation Results:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")

        # Plot actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, y_pred, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--')
        plt.xlabel('Actual Position')
        plt.ylabel('Predicted Position')
        plt.title('Actual vs Predicted Finishing Positions')
        plt.savefig('actual_vs_predicted.png')

        # Plot feature importances if available
        if self.feature_importances is not None:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=self.feature_importances.head(10))
            plt.title('Top 10 Feature Importances')
            plt.tight_layout()
            plt.savefig('feature_importances.png')

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
        -----------
        X : DataFrame
            Feature matrix for prediction

        Returns:
        --------
        array
            Predicted finishing positions
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        return self.best_model.predict(X)

    def save_model(self, filename='f1_prediction_model.joblib'):
        """
        Save the trained model to a file.

        Parameters:
        -----------
        filename : str
            Name of the file to save the model to

        Returns:
        --------
        str
            Path to the saved model file
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        joblib.dump(self.best_model, filename)
        print(f"Model saved to {filename}")
        return filename

    def load_model(self, filename='f1_prediction_model.joblib'):
        """
        Load a trained model from a file.

        Parameters:
        -----------
        filename : str
            Name of the file to load the model from

        Returns:
        --------
        self
            Model instance with loaded model
        """
        self.best_model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        return self

    def describe_relationships(self):
        """
        Describe the relationships between entities in the F1 prediction model.

        Returns:
        --------
        str
            A string describing the relationships between entities
        """
        relationships = """
## Relationships

- **Driver** belongs to a **Team**
- **RaceResult** references a **Driver** and a **Team**
- **LapTime** references a **Driver**
- **PitStop** references a **Driver**
- **QualifyingResult** references a **Driver** and a **Team**
- **DriverStanding** references a **Driver** and a **Team**
- **ConstructorResult** references a **Team** and a **Driver**
- **RaceStatus** references a **Driver**
"""
        return relationships


if __name__ == "__main__":
    # Create an instance of the F1PredictionModel
    model = F1PredictionModel(data_dir=".")

    # Display model relationships
    print(model.describe_relationships())

    # Load data
    print("Loading data...")
    data = model.load_data()

    # Prepare features
    print("Preparing features...")
    X, y, df = model.prepare_features(data)

    # Set X and y in the model
    model.X = X
    model.y = y

    # Train the model
    print("Training model...")
    model.train()

    # Evaluate the model
    print("Evaluating model...")
    metrics = model.evaluate()

    # Save the model
    model.save_model()

    print("\nModel training and evaluation complete!")
    print("You can now use this model to predict F1 race outcomes.")
    print("Example usage:")
    print("    model = F1PredictionModel()")
    print("    model.load_model('f1_prediction_model.joblib')")
    print("    predictions = model.predict(new_data)")
