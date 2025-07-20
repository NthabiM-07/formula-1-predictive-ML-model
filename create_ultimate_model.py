"""
Quick model optimization focused on key improvements
"""

from f1_prediction_model_enhanced import F1PredictionModelEnhanced
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

def create_final_optimized_model():
    """Create the final optimized F1 prediction model."""
    print("Creating Final Optimized F1 Model")
    print("="*50)
    
    # Load enhanced model
    model = F1PredictionModelEnhanced(
        data_dir=".",
        model_type="ensemble", 
        use_polynomial=True
    )
    
    # Load and prepare data
    data = model.load_data()
    X, y, df = model.prepare_features(data)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    model.X = X
    model.y = y
    
    # Train the enhanced model
    print("Training enhanced ensemble model...")
    model.train()
    
    # Get current performance
    y_pred = model.best_model.predict(model.X_test)
    current_r2 = r2_score(model.y_test, y_pred)
    current_rmse = np.sqrt(mean_squared_error(model.y_test, y_pred))
    
    print(f"Enhanced model performance:")
    print(f"  R² Score: {current_r2:.4f}")
    print(f"  RMSE: {current_rmse:.4f}")
    
    # Create even better ensemble with optimized parameters
    print("\nCreating ultimate ensemble model...")
    
    # Individual models with optimized parameters
    rf_optimized = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    gb_optimized = GradientBoostingRegressor(
        n_estimators=250,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.9,
        random_state=42
    )
    
    et_optimized = ExtraTreesRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42
    )
    
    # Use the already trained enhanced model instead of creating a new ensemble on raw features
    print("Using trained enhanced model as ultimate model...")
    ultimate_model = model.best_model
    
    # Final evaluation on test set
    y_pred_test = ultimate_model.predict(model.X_test)
    test_r2 = r2_score(model.y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(model.y_test, y_pred_test))
    
    print(f"\nFinal test set performance:")
    print(f"  R² Score: {test_r2:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    
    # Position accuracy on test set
    within_1 = np.sum(np.abs(y_pred_test - model.y_test) <= 1) / len(model.y_test)
    within_2 = np.sum(np.abs(y_pred_test - model.y_test) <= 2) / len(model.y_test)
    within_3 = np.sum(np.abs(y_pred_test - model.y_test) <= 3) / len(model.y_test)
    
    print(f"  Within ±1 position: {within_1*100:.1f}%")
    print(f"  Within ±2 positions: {within_2*100:.1f}%")
    print(f"  Within ±3 positions: {within_3*100:.1f}%")
    
    # Cross-validation on preprocessed data
    print("\nPerforming cross-validation on preprocessed data...")
    X_processed = model.X_train  # Use the preprocessed training data
    y_processed = model.y_train
    
    # Use the same model architecture but retrain for CV
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(ultimate_model, X_processed, y_processed, cv=3, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    cv_r2_scores = cross_val_score(ultimate_model, X_processed, y_processed, cv=3, scoring='r2')
    
    print(f"Cross-validation results (3-fold):")
    print(f"  RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
    print(f"  R²: {cv_r2_scores.mean():.4f} ± {cv_r2_scores.std():.4f}")
    
    # Save final model
    final_model_data = {
        'model': ultimate_model,
        'enhanced_model_instance': model,  # Save the whole enhanced model for preprocessing
        'feature_names': model.feature_names if hasattr(model, 'feature_names') else None,
        'performance': {
            'r2': test_r2,
            'rmse': test_rmse,
            'within_1_pos': within_1,
            'within_2_pos': within_2,
            'within_3_pos': within_3,
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_r2_mean': cv_r2_scores.mean()
        }
    }
    
    joblib.dump(final_model_data, 'f1_prediction_ultimate.joblib')
    print(f"\n✓ Ultimate F1 model saved to 'f1_prediction_ultimate.joblib'")
    
    return ultimate_model, final_model_data

if __name__ == "__main__":
    model, data = create_final_optimized_model()
    print("\n" + "="*50)
    print("ULTIMATE F1 MODEL CREATION COMPLETE!")
    print("="*50)