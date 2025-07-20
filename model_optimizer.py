"""
F1 Model Optimizer
==================

This module provides advanced model optimization techniques including:
1. Automated hyperparameter optimization
2. Feature importance analysis and selection
3. Cross-validation model comparison
4. Performance monitoring and validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    RandomizedSearchCV, cross_val_score, 
    learning_curve, validation_curve
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    ExtraTreesRegressor, AdaBoostRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.inspection import permutation_importance
import joblib
import warnings
warnings.filterwarnings('ignore')

from f1_prediction_model_enhanced import F1PredictionModelEnhanced

class F1ModelOptimizer:
    """
    Advanced optimizer for F1 prediction models with comprehensive analysis tools.
    """
    
    def __init__(self, data_dir="."):
        self.data_dir = data_dir
        self.base_model = None
        self.optimized_models = {}
        self.feature_analysis = {}
        self.performance_history = {}
        
    def load_base_model(self):
        """Load the enhanced model as a base."""
        self.base_model = F1PredictionModelEnhanced(
            data_dir=self.data_dir,
            model_type="ensemble",
            use_polynomial=True
        )
        
        # Load and prepare data
        data = self.base_model.load_data()
        X, y, df = self.base_model.prepare_features(data)
        
        self.base_model.X = X
        self.base_model.y = y
        
        print(f"Base model loaded with {X.shape[1]} features and {X.shape[0]} samples")
        return self.base_model
        
    def analyze_feature_importance(self, top_k=20):
        """Comprehensive feature importance analysis."""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        if self.base_model is None:
            self.load_base_model()
            
        # Train base model if not already trained
        if not hasattr(self.base_model, 'best_model') or self.base_model.best_model is None:
            self.base_model.train()
            
        X_train = self.base_model.X_train
        X_test = self.base_model.X_test
        y_train = self.base_model.y_train
        y_test = self.base_model.y_test
        
        # Method 1: Tree-based feature importance
        rf_temp = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_temp.fit(X_train, y_train)
        
        # Get feature names after preprocessing
        feature_names = self._get_feature_names(X_train)
        
        tree_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_temp.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Method 2: Permutation importance
        perm_importance = permutation_importance(
            rf_temp, X_test, y_test, n_repeats=10, random_state=42
        )
        
        perm_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        }).sort_values('importance', ascending=False)
        
        # Method 3: Correlation with target
        if hasattr(X_train, 'toarray'):  # Handle sparse matrices
            X_train_dense = X_train.toarray()
        else:
            X_train_dense = X_train
            
        correlations = []
        for i, col in enumerate(feature_names):
            try:
                corr = np.corrcoef(X_train_dense[:, i], y_train)[0, 1]
                correlations.append(abs(corr) if not np.isnan(corr) else 0)
            except:
                correlations.append(0)
                
        corr_importance = pd.DataFrame({
            'feature': feature_names,
            'correlation': correlations
        }).sort_values('correlation', ascending=False)
        
        # Store results
        self.feature_analysis = {
            'tree_importance': tree_importance,
            'permutation_importance': perm_importance_df,
            'correlation_importance': corr_importance
        }
        
        # Print top features
        print(f"\nTop {top_k} Features by Tree Importance:")
        for i, (_, row) in enumerate(tree_importance.head(top_k).iterrows(), 1):
            print(f"  {i:2d}. {row['feature']:<30} {row['importance']:.4f}")
            
        print(f"\nTop {top_k} Features by Permutation Importance:")
        for i, (_, row) in enumerate(perm_importance_df.head(top_k).iterrows(), 1):
            print(f"  {i:2d}. {row['feature']:<30} {row['importance']:.4f} ± {row['std']:.4f}")
            
        # Create visualization
        self._visualize_feature_importance(top_k)
        
        return self.feature_analysis
        
    def _get_feature_names(self, X):
        """Extract feature names from the preprocessed data."""
        if hasattr(X, 'columns'):
            return list(X.columns)
        else:
            # For transformed data from pipeline, try to get names from base model
            if hasattr(self.base_model, 'feature_names') and self.base_model.feature_names:
                # This gets the pre-transformation feature names
                base_features = self.base_model.feature_names
                # For polynomial features, we'll have more features than original
                n_features = X.shape[1]
                if len(base_features) < n_features:
                    # Add polynomial interaction names
                    all_names = base_features.copy()
                    for i in range(len(base_features), n_features):
                        all_names.append(f'poly_feature_{i}')
                    return all_names[:n_features]
                else:
                    return base_features[:n_features]
            else:
                # Create generic names
                return [f'feature_{i}' for i in range(X.shape[1])]
            
    def _visualize_feature_importance(self, top_k=15):
        """Create feature importance visualization."""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Tree importance
        top_tree = self.feature_analysis['tree_importance'].head(top_k)
        axes[0].barh(range(len(top_tree)), top_tree['importance'])
        axes[0].set_yticks(range(len(top_tree)))
        axes[0].set_yticklabels(top_tree['feature'], fontsize=8)
        axes[0].set_xlabel('Importance')
        axes[0].set_title('Tree-based Feature Importance')
        axes[0].invert_yaxis()
        
        # Permutation importance
        top_perm = self.feature_analysis['permutation_importance'].head(top_k)
        axes[1].barh(range(len(top_perm)), top_perm['importance'], 
                    xerr=top_perm['std'], capsize=3)
        axes[1].set_yticks(range(len(top_perm)))
        axes[1].set_yticklabels(top_perm['feature'], fontsize=8)
        axes[1].set_xlabel('Importance')
        axes[1].set_title('Permutation Feature Importance')
        axes[1].invert_yaxis()
        
        # Correlation importance
        top_corr = self.feature_analysis['correlation_importance'].head(top_k)
        axes[2].barh(range(len(top_corr)), top_corr['correlation'])
        axes[2].set_yticks(range(len(top_corr)))
        axes[2].set_yticklabels(top_corr['feature'], fontsize=8)
        axes[2].set_xlabel('Correlation with Target')
        axes[2].set_title('Correlation-based Importance')
        axes[2].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Feature importance analysis saved to 'feature_importance_analysis.png'")
        
    def optimize_hyperparameters(self, n_iter=50, cv=3):
        """Advanced hyperparameter optimization using RandomizedSearchCV."""
        print("\n" + "="*60)
        print("HYPERPARAMETER OPTIMIZATION")
        print("="*60)
        
        if self.base_model is None:
            self.load_base_model()
            
        X = self.base_model.X
        y = self.base_model.y
        
        # Define parameter distributions for different algorithms
        param_distributions = {
            'random_forest': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.15, 0.2],
                'max_depth': [3, 5, 7, 9],
                'subsample': [0.8, 0.9, 1.0],
                'min_samples_split': [2, 5, 10]
            },
            'extra_trees': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
        
        # Models to optimize
        models = {
            'random_forest': RandomForestRegressor(random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'extra_trees': ExtraTreesRegressor(random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nOptimizing {name}...")
            
            # Create RandomizedSearchCV
            random_search = RandomizedSearchCV(
                model,
                param_distributions[name],
                n_iter=n_iter,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
            
            # Fit the search
            random_search.fit(X, y)
            
            # Store results
            results[name] = {
                'best_estimator': random_search.best_estimator_,
                'best_params': random_search.best_params_,
                'best_score': -random_search.best_score_,
                'cv_results': random_search.cv_results_
            }
            
            print(f"  Best RMSE: {np.sqrt(-random_search.best_score_):.4f}")
            print(f"  Best params: {random_search.best_params_}")
            
        self.optimized_models = results
        return results
        
    def compare_models(self, cv=5):
        """Compare different model architectures."""
        print("\n" + "="*60)
        print("MODEL ARCHITECTURE COMPARISON")
        print("="*60)
        
        if self.base_model is None:
            self.load_base_model()
            
        X = self.base_model.X
        y = self.base_model.y
        
        # Define models to compare
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
            'Extra Trees': ExtraTreesRegressor(n_estimators=200, random_state=42),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'ElasticNet': ElasticNet(alpha=1.0),
            'SVR': SVR(kernel='rbf', C=1.0)
        }
        
        # Store results
        comparison_results = {}
        
        for name, model in models.items():
            print(f"\nEvaluating {name}...")
            
            try:
                # Cross-validation scores
                cv_scores = cross_val_score(
                    model, X, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1
                )
                rmse_scores = np.sqrt(-cv_scores)
                
                # R² scores
                r2_scores = cross_val_score(
                    model, X, y, cv=cv, scoring='r2', n_jobs=-1
                )
                
                comparison_results[name] = {
                    'rmse_mean': rmse_scores.mean(),
                    'rmse_std': rmse_scores.std(),
                    'r2_mean': r2_scores.mean(),
                    'r2_std': r2_scores.std()
                }
                
                print(f"  RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
                print(f"  R²: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
                
            except Exception as e:
                print(f"  Error with {name}: {e}")
                comparison_results[name] = {
                    'rmse_mean': np.inf, 'rmse_std': 0,
                    'r2_mean': -np.inf, 'r2_std': 0
                }
        
        # Create comparison visualization
        self._visualize_model_comparison(comparison_results)
        
        return comparison_results
        
    def _visualize_model_comparison(self, results):
        """Visualize model comparison results."""
        models = list(results.keys())
        rmse_means = [results[m]['rmse_mean'] for m in models]
        rmse_stds = [results[m]['rmse_std'] for m in models]
        r2_means = [results[m]['r2_mean'] for m in models]
        r2_stds = [results[m]['r2_std'] for m in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # RMSE comparison
        x_pos = np.arange(len(models))
        ax1.bar(x_pos, rmse_means, yerr=rmse_stds, capsize=5, alpha=0.7)
        ax1.set_xlabel('Models')
        ax1.set_ylabel('RMSE')
        ax1.set_title('Model Comparison - RMSE')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # R² comparison
        ax2.bar(x_pos, r2_means, yerr=r2_stds, capsize=5, alpha=0.7, color='green')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('R² Score')
        ax2.set_title('Model Comparison - R² Score')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_architecture_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Model comparison saved to 'model_architecture_comparison.png'")
        
    def analyze_learning_curves(self, model_name='random_forest'):
        """Analyze learning curves to understand model performance vs dataset size."""
        print("\n" + "="*60)
        print("LEARNING CURVE ANALYSIS")
        print("="*60)
        
        if self.base_model is None:
            self.load_base_model()
            
        X = self.base_model.X
        y = self.base_model.y
        
        # Use optimized model if available
        if model_name in self.optimized_models:
            model = self.optimized_models[model_name]['best_estimator']
        else:
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            
        # Generate learning curve
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=3, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='neg_mean_squared_error'
        )
        
        # Convert to RMSE
        train_rmse = np.sqrt(-train_scores)
        val_rmse = np.sqrt(-val_scores)
        
        # Calculate means and standard deviations
        train_rmse_mean = train_rmse.mean(axis=1)
        train_rmse_std = train_rmse.std(axis=1)
        val_rmse_mean = val_rmse.mean(axis=1)
        val_rmse_std = val_rmse.std(axis=1)
        
        # Plot learning curves
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_sizes, train_rmse_mean, 'o-', color='blue', label='Training RMSE')
        plt.fill_between(train_sizes, train_rmse_mean - train_rmse_std,
                        train_rmse_mean + train_rmse_std, alpha=0.1, color='blue')
        plt.plot(train_sizes, val_rmse_mean, 'o-', color='red', label='Validation RMSE')
        plt.fill_between(train_sizes, val_rmse_mean - val_rmse_std,
                        val_rmse_mean + val_rmse_std, alpha=0.1, color='red')
        plt.xlabel('Training Set Size')
        plt.ylabel('RMSE')
        plt.title('Learning Curve - RMSE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Analyze overfitting
        overfitting = train_rmse_mean - val_rmse_mean
        plt.subplot(1, 2, 2)
        plt.plot(train_sizes, overfitting, 'o-', color='orange', label='Overfitting Gap')
        plt.xlabel('Training Set Size')
        plt.ylabel('Training RMSE - Validation RMSE')
        plt.title('Overfitting Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('learning_curve_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✓ Learning curve analysis saved to 'learning_curve_analysis.png'")
        
        return {
            'train_sizes': train_sizes,
            'train_rmse_mean': train_rmse_mean,
            'val_rmse_mean': val_rmse_mean,
            'overfitting': overfitting
        }
        
    def create_final_optimized_model(self):
        """Create the final optimized model using best practices."""
        print("\n" + "="*60)
        print("CREATING FINAL OPTIMIZED MODEL")
        print("="*60)
        
        if self.base_model is None:
            self.load_base_model()
            
        # Use the best performing model from optimization
        if self.optimized_models:
            best_model_name = min(self.optimized_models.keys(), 
                                key=lambda x: self.optimized_models[x]['best_score'])
            best_model = self.optimized_models[best_model_name]['best_estimator']
            print(f"Using optimized {best_model_name} as final model")
        else:
            # Fallback to ensemble
            from sklearn.ensemble import VotingRegressor
            rf = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
            gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
            et = ExtraTreesRegressor(n_estimators=200, random_state=42)
            
            best_model = VotingRegressor([
                ('rf', rf), ('gb', gb), ('et', et)
            ])
            print("Using ensemble model as final model")
            
        # Train final model
        best_model.fit(self.base_model.X, self.base_model.y)
        
        # Evaluate
        y_pred = best_model.predict(self.base_model.X)
        final_r2 = r2_score(self.base_model.y, y_pred)
        final_rmse = np.sqrt(mean_squared_error(self.base_model.y, y_pred))
        
        print(f"Final model performance:")
        print(f"  R² Score: {final_r2:.4f}")
        print(f"  RMSE: {final_rmse:.4f}")
        
        # Save final model
        joblib.dump(best_model, 'f1_prediction_final_optimized.joblib')
        print(f"✓ Final optimized model saved to 'f1_prediction_final_optimized.joblib'")
        
        return best_model
        
def main():
    """Run complete model optimization pipeline."""
    print("F1 Model Optimization Pipeline")
    print("="*50)
    
    # Initialize optimizer
    optimizer = F1ModelOptimizer(data_dir=".")
    
    # Load base model
    optimizer.load_base_model()
    
    # Analyze feature importance
    optimizer.analyze_feature_importance(top_k=15)
    
    # Optimize hyperparameters
    optimizer.optimize_hyperparameters(n_iter=30, cv=3)
    
    # Compare different model architectures
    optimizer.compare_models(cv=3)
    
    # Analyze learning curves
    optimizer.analyze_learning_curves()
    
    # Create final optimized model
    final_model = optimizer.create_final_optimized_model()
    
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE!")
    print("="*50)
    print("Generated files:")
    print("  • feature_importance_analysis.png")
    print("  • model_architecture_comparison.png") 
    print("  • learning_curve_analysis.png")
    print("  • f1_prediction_final_optimized.joblib")

if __name__ == "__main__":
    main()