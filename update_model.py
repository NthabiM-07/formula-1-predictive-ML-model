"""
F1 Model Update Manager
======================

This script manages automatic model updates after each race:
1. Loads new race data
2. Updates the training dataset
3. Retrains the model
4. Validates performance
5. Manages model versioning

Usage:
------
python update_model.py --race "Hungary GP" --round 13
python update_model.py --auto-update
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import datetime
import shutil
import argparse
from f1_prediction_model import F1PredictionModel
from sklearn.metrics import mean_squared_error, r2_score

class F1ModelUpdater:
    def __init__(self, data_dir=".", models_dir="models", config_file="model_config.json"):
        """Initialize the model updater."""
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.config_file = config_file
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Load or create configuration
        self.config = self.load_config()
        
    def load_config(self):
        """Load model configuration or create default."""
        default_config = {
            "last_updated": None,
            "last_race_round": 0,
            "model_version": "1.0.0",
            "performance_history": [],
            "data_sources": [],
            "update_threshold": 0.05,  # Minimum improvement to keep new model
            "backup_count": 5  # Number of model backups to keep
        }
        
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                # Merge with defaults for any missing keys
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        else:
            return default_config
    
    def save_config(self):
        """Save configuration to file."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
    
    def detect_new_races(self):
        """Detect new race folders that haven't been processed."""
        # Find all GP folders
        gp_folders = [d for d in os.listdir(self.data_dir) 
                     if os.path.isdir(os.path.join(self.data_dir, d)) and "GP" in d]
        
        new_races = []
        for folder in gp_folders:
            # Extract round number if available
            round_num = self.extract_round_number(folder)
            if round_num and round_num > self.config["last_race_round"]:
                new_races.append((folder, round_num))
        
        # Sort by round number
        new_races.sort(key=lambda x: x[1])
        return new_races
    
    def extract_round_number(self, folder_name):
        """Extract round number from folder name."""
        # Look for files with round numbers
        folder_path = os.path.join(self.data_dir, folder_name)
        try:
            files = os.listdir(folder_path)
            for file in files:
                if "round_" in file.lower():
                    parts = file.lower().split("round_")
                    if len(parts) > 1:
                        round_str = parts[1].split(".")[0].split("_")[0]
                        return int(round_str)
        except:
            pass
        return None
    
    def backup_current_model(self):
        """Create a backup of the current model."""
        if os.path.exists("f1_prediction_model.joblib"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version = self.config["model_version"]
            backup_name = f"f1_model_v{version}_{timestamp}.joblib"
            backup_path = os.path.join(self.models_dir, backup_name)
            
            shutil.copy2("f1_prediction_model.joblib", backup_path)
            print(f"✓ Backed up current model to {backup_path}")
            
            # Clean up old backups
            self.cleanup_old_backups()
            
            return backup_path
        return None
    
    def cleanup_old_backups(self):
        """Remove old model backups, keeping only the most recent ones."""
        backup_files = [f for f in os.listdir(self.models_dir) if f.startswith("f1_model_v")]
        backup_files.sort(reverse=True)  # Newest first
        
        # Remove excess backups
        for old_backup in backup_files[self.config["backup_count"]:]:
            old_path = os.path.join(self.models_dir, old_backup)
            os.remove(old_path)
            print(f"🗑️ Removed old backup: {old_backup}")
    
    def evaluate_model_performance(self, model, X, y):
        """Evaluate model performance and return metrics."""
        predictions = model.predict(X)
        
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, predictions)
        mae = np.mean(np.abs(predictions - y))
        
        return {
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "mae": mae,
            "timestamp": datetime.now(),
            "data_size": len(y)
        }
    
    def should_update_model(self, new_performance, old_performance):
        """Determine if new model is significantly better."""
        if not old_performance:
            return True  # No previous model
        
        # Check if R² improved by threshold
        r2_improvement = new_performance["r2"] - old_performance["r2"]
        
        # Also consider RMSE improvement
        rmse_improvement = old_performance["rmse"] - new_performance["rmse"]
        
        print(f"Performance comparison:")
        print(f"  R² change: {r2_improvement:+.4f}")
        print(f"  RMSE change: {rmse_improvement:+.2f}")
        
        # Update if R² improved by threshold OR RMSE improved significantly
        return (r2_improvement >= self.config["update_threshold"] or 
                rmse_improvement >= 1.0)
    
    def update_model_incremental(self, new_race_folders):
        """Update model incrementally with new race data."""
        print(f"\n🔄 Updating model with {len(new_race_folders)} new races...")
        
        # Backup current model
        backup_path = self.backup_current_model()
        
        # Load current data and model
        f1_model = F1PredictionModel(data_dir=self.data_dir)
        
        # Get current performance baseline
        try:
            current_model = joblib.load("f1_prediction_model.joblib")
            all_data = f1_model.load_data()
            X_current, y_current, _ = f1_model.prepare_features(all_data)
            old_performance = self.evaluate_model_performance(current_model, X_current, y_current)
            print(f"📊 Current model performance: R² = {old_performance['r2']:.4f}, RMSE = {old_performance['rmse']:.2f}")
        except FileNotFoundError:
            old_performance = None
            print("📊 No existing model found")
        
        # Load data including new races
        race_folders = [folder for folder, _ in new_race_folders]
        all_data = f1_model.load_data()  # This loads all available data
        X_new, y_new, df_new = f1_model.prepare_features(all_data)
        
        print(f"📈 Updated dataset: {X_new.shape[0]} samples (+{X_new.shape[0] - (X_current.shape[0] if old_performance else 0)})")
        
        # Train new model
        f1_model.X = X_new
        f1_model.y = y_new
        print("🏋️ Training new model...")
        f1_model.train()
        
        # Evaluate new model
        new_performance = self.evaluate_model_performance(f1_model.best_model, X_new, y_new)
        print(f"📊 New model performance: R² = {new_performance['r2']:.4f}, RMSE = {new_performance['rmse']:.2f}")
        
        # Decide whether to keep new model
        if self.should_update_model(new_performance, old_performance):
            # Save new model
            f1_model.save_model()
            
            # Update configuration
            max_round = max([round_num for _, round_num in new_race_folders])
            self.config["last_race_round"] = max_round
            self.config["last_updated"] = datetime.now()
            
            # Increment version
            version_parts = self.config["model_version"].split(".")
            version_parts[-1] = str(int(version_parts[-1]) + 1)
            self.config["model_version"] = ".".join(version_parts)
            
            # Add performance to history
            self.config["performance_history"].append(new_performance)
            
            # Add new data sources
            for folder, round_num in new_race_folders:
                self.config["data_sources"].append({
                    "race": folder,
                    "round": round_num,
                    "added": datetime.now()
                })
            
            self.save_config()
            
            print(f"✅ Model updated successfully!")
            print(f"   New version: {self.config['model_version']}")
            print(f"   Last race: Round {max_round}")
            
            return True
        else:
            # Restore backup
            if backup_path and os.path.exists(backup_path):
                shutil.copy2(backup_path, "f1_prediction_model.joblib")
                print("↩️ Model performance didn't improve enough. Restored previous model.")
            
            return False
    
    def update_model_full_retrain(self):
        """Perform a full retrain with all available data."""
        print(f"\n🔄 Performing full model retrain...")
        
        # Backup current model
        backup_path = self.backup_current_model()
        
        # Load all available data
        f1_model = F1PredictionModel(data_dir=self.data_dir)
        all_data = f1_model.load_data()
        X, y, df = f1_model.prepare_features(all_data)
        
        print(f"📈 Full dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Train model
        f1_model.X = X
        f1_model.y = y
        print("🏋️ Training model on full dataset...")
        f1_model.train()
        
        # Evaluate
        performance = self.evaluate_model_performance(f1_model.best_model, X, y)
        print(f"📊 Model performance: R² = {performance['r2']:.4f}, RMSE = {performance['rmse']:.2f}")
        
        # Save model
        f1_model.save_model()
        
        # Update configuration
        races = self.detect_new_races()
        if races:
            max_round = max([round_num for _, round_num in races])
            self.config["last_race_round"] = max_round
        
        self.config["last_updated"] = datetime.now()
        self.config["performance_history"].append(performance)
        
        # Increment major version for full retrain
        version_parts = self.config["model_version"].split(".")
        version_parts[0] = str(int(version_parts[0]) + 1)
        version_parts[1] = "0"
        version_parts[2] = "0"
        self.config["model_version"] = ".".join(version_parts)
        
        self.save_config()
        
        print(f"✅ Full retrain completed!")
        print(f"   New version: {self.config['model_version']}")
        
        return True
    
    def add_new_race_data(self, race_name, round_number):
        """Add new race data manually."""
        print(f"📥 Adding race data: {race_name} (Round {round_number})")
        
        # Check if race folder exists
        race_folder = None
        for folder in os.listdir(self.data_dir):
            if race_name.lower() in folder.lower() and os.path.isdir(os.path.join(self.data_dir, folder)):
                race_folder = folder
                break
        
        if not race_folder:
            print(f"❌ Race folder not found for '{race_name}'")
            return False
        
        print(f"✓ Found race folder: {race_folder}")
        
        # Update model with this new race
        return self.update_model_incremental([(race_folder, round_number)])
    
    def auto_update(self):
        """Automatically detect and process new races."""
        print("🔍 Checking for new race data...")
        
        new_races = self.detect_new_races()
        
        if not new_races:
            print("✓ No new races found. Model is up to date.")
            return
        
        print(f"📅 Found {len(new_races)} new races:")
        for race, round_num in new_races:
            print(f"   - {race} (Round {round_num})")
        
        # Update model incrementally
        success = self.update_model_incremental(new_races)
        
        if success:
            print(f"\n🎉 Model successfully updated with new race data!")
            self.print_status()
        else:
            print(f"\n⚠️ Model update completed but performance didn't improve significantly.")
    
    def print_status(self):
        """Print current model status."""
        print(f"\n📊 Model Status:")
        print(f"   Version: {self.config['model_version']}")
        print(f"   Last Updated: {self.config['last_updated']}")
        print(f"   Last Race Round: {self.config['last_race_round']}")
        print(f"   Total Races: {len(self.config['data_sources'])}")
        
        if self.config["performance_history"]:
            latest = self.config["performance_history"][-1]
            print(f"   Current Performance:")
            print(f"     - R² Score: {latest['r2']:.4f}")
            print(f"     - RMSE: {latest['rmse']:.2f}")
            print(f"     - Data Size: {latest['data_size']} samples")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='F1 Model Update Manager')
    parser.add_argument('--race', type=str, help='Race name to add (e.g., "Hungary GP")')
    parser.add_argument('--round', type=int, help='Round number')
    parser.add_argument('--auto-update', action='store_true', help='Automatically detect and add new races')
    parser.add_argument('--full-retrain', action='store_true', help='Perform full model retrain')
    parser.add_argument('--status', action='store_true', help='Show current model status')
    
    args = parser.parse_args()
    
    # Initialize updater
    updater = F1ModelUpdater()
    
    if args.status:
        updater.print_status()
    elif args.race and args.round:
        updater.add_new_race_data(args.race, args.round)
    elif args.auto_update:
        updater.auto_update()
    elif args.full_retrain:
        updater.update_model_full_retrain()
    else:
        # Default: auto-update
        print("F1 Model Update Manager")
        print("=" * 30)
        updater.auto_update()

if __name__ == "__main__":
    main()
