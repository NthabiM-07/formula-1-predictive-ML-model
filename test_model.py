"""
F1 Model Testing Suite
=====================

This script provides comprehensive testing for the F1 prediction model,
including performance evaluation, cross-validation, and prediction testing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import joblib
from f1_prediction_model import F1PredictionModel

# Set matplotlib backend to avoid GUI issues
plt.switch_backend('Agg')

class F1ModelTester:
    def __init__(self, model_path='f1_prediction_model.joblib'):
        """Initialize the model tester."""
        self.model_path = model_path
        self.model = None
        self.data = None
        self.X = None
        self.y = None
        
    def load_model_and_data(self):
        """Load the trained model and data for testing."""
        print("Loading model and data...")
        
        # Create model instance and load data
        f1_model = F1PredictionModel(data_dir=".")
        self.data = f1_model.load_data()
        self.X, self.y, self.df = f1_model.prepare_features(self.data)
        
        # Load trained model
        try:
            self.model = joblib.load(self.model_path)
            print(f"✓ Model loaded from {self.model_path}")
        except FileNotFoundError:
            print(f"❌ Model file {self.model_path} not found. Training new model...")
            f1_model.X = self.X
            f1_model.y = self.y
            f1_model.train()
            self.model = f1_model.best_model
            f1_model.save_model(self.model_path)
            print(f"✓ New model trained and saved to {self.model_path}")
        
        print(f"Dataset shape: {self.X.shape}")
        print(f"Features: {list(self.X.columns)}")
        
    def test_basic_predictions(self):
        """Test basic prediction functionality."""
        print("\n" + "="*50)
        print("1. BASIC PREDICTION TEST")
        print("="*50)
        
        # Make predictions on full dataset
        predictions = self.model.predict(self.X)
        
        # Calculate metrics
        mse = mean_squared_error(self.y, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y, predictions)
        r2 = r2_score(self.y, predictions)
        
        print(f"Full Dataset Performance:")
        print(f"  - R² Score: {r2:.4f}")
        print(f"  - RMSE: {rmse:.2f} positions")
        print(f"  - MAE: {mae:.2f} positions")
        
        # Create predictions vs actual plot
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y, predictions, alpha=0.7, color='blue')
        plt.plot([self.y.min(), self.y.max()], [self.y.min(), self.y.max()], 'r--', lw=2)
        plt.xlabel('Actual Position')
        plt.ylabel('Predicted Position')
        plt.title('Actual vs Predicted Positions (Full Dataset)')
        plt.grid(True, alpha=0.3)
        plt.savefig('test_predictions_full.png', dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to 'test_predictions_full.png'")
        
        return predictions
        
    def test_position_accuracy(self, predictions):
        """Test position-based accuracy metrics."""
        print("\n" + "="*50)
        print("2. POSITION ACCURACY TEST")
        print("="*50)
        
        # Calculate position-based accuracy
        exact_matches = np.sum(np.round(predictions) == self.y)
        within_1 = np.sum(np.abs(predictions - self.y) <= 1)
        within_2 = np.sum(np.abs(predictions - self.y) <= 2)
        within_3 = np.sum(np.abs(predictions - self.y) <= 3)
        
        total = len(self.y)
        
        print(f"Position Accuracy:")
        print(f"  - Exact position: {exact_matches}/{total} ({exact_matches/total*100:.1f}%)")
        print(f"  - Within ±1 position: {within_1}/{total} ({within_1/total*100:.1f}%)")
        print(f"  - Within ±2 positions: {within_2}/{total} ({within_2/total*100:.1f}%)")
        print(f"  - Within ±3 positions: {within_3}/{total} ({within_3/total*100:.1f}%)")
        
        # Test top positions accuracy (top 3, top 5, top 10)
        print(f"\nTop Position Predictions:")
        for top_n in [3, 5, 10]:
            actual_top = set(self.y[self.y <= top_n].index)
            pred_top = set(np.argsort(predictions)[:top_n])
            overlap = len(actual_top.intersection(pred_top))
            print(f"  - Top {top_n} accuracy: {overlap}/{top_n} ({overlap/top_n*100:.1f}%)")
    
    def test_cross_validation(self):
        """Perform cross-validation testing."""
        print("\n" + "="*50)
        print("3. CROSS-VALIDATION TEST")
        print("="*50)
        
        # Time series split for temporal validation
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = cross_val_score(self.model, self.X, self.y, 
                                   cv=tscv, scoring='neg_mean_squared_error')
        
        rmse_scores = np.sqrt(-cv_scores)
        
        print(f"Time Series Cross-Validation (3 folds):")
        print(f"  - RMSE scores: {rmse_scores}")
        print(f"  - Mean RMSE: {rmse_scores.mean():.3f} ± {rmse_scores.std():.3f}")
        
        # R² cross-validation
        r2_scores = cross_val_score(self.model, self.X, self.y, 
                                   cv=tscv, scoring='r2')
        print(f"  - R² scores: {r2_scores}")
        print(f"  - Mean R²: {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")
        
    def test_feature_importance(self):
        """Test and visualize feature importance."""
        print("\n" + "="*50)
        print("4. FEATURE IMPORTANCE TEST")
        print("="*50)
        
        if hasattr(self.model.named_steps['model'], 'feature_importances_'):
            # Get feature names after preprocessing
            preprocessor = self.model.named_steps['preprocessor']
            feature_names = []
            
            # Get numerical features
            numerical_features = [col for col in self.X.columns if col not in ['Team', 'Team_x']]
            feature_names.extend(numerical_features)
            
            # Get categorical features
            categorical_features = ['Team'] if 'Team' in self.X.columns else ['Team_x'] if 'Team_x' in self.X.columns else []
            if categorical_features:
                # Find categorical transformer
                cat_transformers = [t for t in preprocessor.transformers_ if t[0] == 'cat']
                if cat_transformers:
                    cat_transformer = cat_transformers[0][1]
                    if hasattr(cat_transformer.named_steps['onehot'], 'categories_'):
                        categories = cat_transformer.named_steps['onehot'].categories_[0]
                        cat_features = [f"{categorical_features[0]}_{cat}" for cat in categories]
                        feature_names.extend(cat_features)
            
            importances = self.model.named_steps['model'].feature_importances_
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names[:len(importances)],
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print("Feature Importance Rankings:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                print(f"  {i:2d}. {row['Feature']:<25} {row['Importance']:.4f}")
            
            # Plot feature importance
            plt.figure(figsize=(12, 6))
            sns.barplot(data=importance_df.head(10), x='Importance', y='Feature')
            plt.title('Top 10 Feature Importances')
            plt.tight_layout()
            plt.savefig('test_feature_importance.png', dpi=300, bbox_inches='tight')
            print(f"✓ Feature importance plot saved to 'test_feature_importance.png'")
        else:
            print("Model does not provide feature importance information")
    
    def test_new_data_prediction(self):
        """Test prediction on new/hypothetical data."""
        print("\n" + "="*50)
        print("5. NEW DATA PREDICTION TEST")
        print("="*50)
        
        # Create test scenarios
        test_scenarios = [
            {
                'name': 'Perfect Qualifying Order',
                'data': pd.DataFrame({
                    'Position_x': [1, 2, 3, 4, 5],
                    'best_qualifying_time': [64.0, 64.2, 64.4, 64.6, 64.8],
                    'GridPosition': [1, 2, 3, 4, 5],
                    'grid_position_diff': [0, 0, 0, 0, 0],
                    'Team': ['McLaren', 'Ferrari', 'Red Bull Racing', 'Mercedes', 'Aston Martin']
                })
            },
            {
                'name': 'Mixed Grid Scenario',
                'data': pd.DataFrame({
                    'Position_x': [1, 5, 3, 2, 4],
                    'best_qualifying_time': [64.0, 65.0, 64.5, 64.2, 64.7],
                    'GridPosition': [1, 5, 3, 2, 4],
                    'grid_position_diff': [0, 0, 0, 0, 0],
                    'Team': ['McLaren', 'Williams', 'Ferrari', 'Red Bull Racing', 'Mercedes']
                })
            }
        ]
        
        for scenario in test_scenarios:
            print(f"\nScenario: {scenario['name']}")
            test_data = scenario['data'].copy()
            
            # Add missing features with default values
            for col in self.X.columns:
                if col not in test_data.columns:
                    if col in ['prev_race_position', 'prev_qualifying_position', 
                              'Position_prev_standings', 'Points_prev_standings']:
                        test_data[col] = np.nan
                    else:
                        test_data[col] = self.X[col].mean()
            
            # Fill NaN values
            test_data = test_data.fillna(test_data.mean(numeric_only=True))
            
            try:
                predictions = self.model.predict(test_data[self.X.columns])
                
                results = pd.DataFrame({
                    'Qualifying_Pos': test_data['Position_x'],
                    'Team': test_data['Team'],
                    'Predicted_Finish': predictions.round(1)
                })
                print(results.to_string(index=False))
                
            except Exception as e:
                print(f"❌ Error in prediction: {e}")
    
    def test_model_robustness(self):
        """Test model robustness with edge cases."""
        print("\n" + "="*50)
        print("6. ROBUSTNESS TEST")
        print("="*50)
        
        # Test with missing values
        print("Testing with missing values...")
        X_missing = self.X.copy()
        X_missing.iloc[0, 1] = np.nan  # Set one value to NaN
        
        try:
            pred_missing = self.model.predict(X_missing)
            print("✓ Model handles missing values correctly")
        except Exception as e:
            print(f"❌ Model fails with missing values: {e}")
        
        # Test with extreme values
        print("Testing with extreme values...")
        X_extreme = self.X.copy()
        X_extreme.iloc[0, 0] = 100  # Extreme qualifying position
        
        try:
            pred_extreme = self.model.predict(X_extreme)
            print("✓ Model handles extreme values")
        except Exception as e:
            print(f"❌ Model fails with extreme values: {e}")
    
    def generate_test_report(self):
        """Generate a comprehensive test report."""
        print("\n" + "="*60)
        print("MODEL TEST REPORT SUMMARY")
        print("="*60)
        
        # Basic model info
        print(f"Model Type: {type(self.model.named_steps['model']).__name__}")
        print(f"Dataset Size: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        print(f"Target Range: {self.y.min():.0f} - {self.y.max():.0f} (positions)")
        
        # Quick performance summary
        predictions = self.model.predict(self.X)
        rmse = np.sqrt(mean_squared_error(self.y, predictions))
        r2 = r2_score(self.y, predictions)
        
        print(f"\nPerformance Summary:")
        print(f"  - R² Score: {r2:.4f}")
        print(f"  - RMSE: {rmse:.2f} positions")
        
        # Recommendations
        print(f"\nRecommendations:")
        if r2 < 0.2:
            print("  ⚠️  Low R² score - consider more data or better features")
        if rmse > 4:
            print("  ⚠️  High RMSE - predictions may be too uncertain")
        if self.X.shape[0] < 100:
            print("  ⚠️  Small dataset - collect more data for better generalization")
        
        print(f"\nTest files generated:")
        print("  - test_predictions_full.png")
        print("  - test_feature_importance.png")

def main():
    """Run the complete test suite."""
    print("F1 Model Testing Suite")
    print("=" * 60)
    
    # Initialize tester
    tester = F1ModelTester()
    
    # Run all tests
    try:
        tester.load_model_and_data()
        predictions = tester.test_basic_predictions()
        tester.test_position_accuracy(predictions)
        tester.test_cross_validation()
        tester.test_feature_importance()
        tester.test_new_data_prediction()
        tester.test_model_robustness()
        tester.generate_test_report()
        
        print(f"\n✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
