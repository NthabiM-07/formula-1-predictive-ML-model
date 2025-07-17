"""
F1 Season Data Manager
=====================

Manages F1 data collection and processing throughout the season.
Handles data validation, cleaning, and preparation for model updates.
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
try:
    import schedule
    import requests
except ImportError:
    schedule = None
    requests = None
import time

class F1DataManager:
    def __init__(self, data_dir=".", season=2025):
        """Initialize the data manager."""
        self.data_dir = data_dir
        self.season = season
        self.race_calendar = self.load_race_calendar()
        
    def load_race_calendar(self):
        """Load or create race calendar for the season."""
        calendar_file = f"race_calendar_{self.season}.json"
        
        # Default F1 2025 calendar (you can update this)
        default_calendar = [
            {"round": 1, "name": "Bahrain GP", "date": "2025-03-16", "location": "Sakhir"},
            {"round": 2, "name": "Saudi Arabia GP", "date": "2025-03-23", "location": "Jeddah"},
            {"round": 3, "name": "Australia GP", "date": "2025-04-06", "location": "Melbourne"},
            {"round": 4, "name": "Japan GP", "date": "2025-04-13", "location": "Suzuka"},
            {"round": 5, "name": "China GP", "date": "2025-04-20", "location": "Shanghai"},
            {"round": 6, "name": "Miami GP", "date": "2025-05-04", "location": "Miami"},
            {"round": 7, "name": "Emilia Romagna GP", "date": "2025-05-18", "location": "Imola"},
            {"round": 8, "name": "Monaco GP", "date": "2025-05-25", "location": "Monaco"},
            {"round": 9, "name": "Spain GP", "date": "2025-06-01", "location": "Barcelona"},
            {"round": 10, "name": "Canada GP", "date": "2025-06-15", "location": "Montreal"},
            {"round": 11, "name": "Austria GP", "date": "2025-06-29", "location": "Spielberg"},
            {"round": 12, "name": "UK GP", "date": "2025-07-06", "location": "Silverstone"},
            {"round": 13, "name": "Hungary GP", "date": "2025-07-20", "location": "Budapest"},
            {"round": 14, "name": "Belgium GP", "date": "2025-07-27", "location": "Spa"},
            {"round": 15, "name": "Netherlands GP", "date": "2025-08-31", "location": "Zandvoort"},
            {"round": 16, "name": "Italy GP", "date": "2025-09-07", "location": "Monza"},
            {"round": 17, "name": "Azerbaijan GP", "date": "2025-09-21", "location": "Baku"},
            {"round": 18, "name": "Singapore GP", "date": "2025-10-05", "location": "Singapore"},
            {"round": 19, "name": "USA GP", "date": "2025-10-19", "location": "Austin"},
            {"round": 20, "name": "Mexico GP", "date": "2025-10-26", "location": "Mexico City"},
            {"round": 21, "name": "Brazil GP", "date": "2025-11-09", "location": "São Paulo"},
            {"round": 22, "name": "Las Vegas GP", "date": "2025-11-22", "location": "Las Vegas"},
            {"round": 23, "name": "Qatar GP", "date": "2025-11-30", "location": "Lusail"},
            {"round": 24, "name": "Abu Dhabi GP", "date": "2025-12-07", "location": "Yas Marina"}
        ]
        
        if os.path.exists(calendar_file):
            with open(calendar_file, 'r') as f:
                return json.load(f)
        else:
            with open(calendar_file, 'w') as f:
                json.dump(default_calendar, f, indent=2)
            return default_calendar
    
    def get_next_race(self):
        """Get the next upcoming race."""
        today = datetime.now().date()
        
        for race in self.race_calendar:
            race_date = datetime.strptime(race["date"], "%Y-%m-%d").date()
            if race_date > today:
                return race
        
        return None
    
    def get_recent_races(self, days=7):
        """Get races that finished in the last N days."""
        cutoff_date = datetime.now().date() - timedelta(days=days)
        recent_races = []
        
        for race in self.race_calendar:
            race_date = datetime.strptime(race["date"], "%Y-%m-%d").date()
            if cutoff_date <= race_date <= datetime.now().date():
                recent_races.append(race)
        
        return recent_races
    
    def validate_race_data(self, race_folder):
        """Validate that race folder contains all required files."""
        required_files = [
            "qualifying_results",
            "race_results",
            "driver_standings",
            "lap_times"
        ]
        
        folder_path = os.path.join(self.data_dir, race_folder)
        if not os.path.exists(folder_path):
            return False, f"Folder {race_folder} not found"
        
        missing_files = []
        for file_type in required_files:
            matching_files = [f for f in os.listdir(folder_path) if file_type in f.lower()]
            if not matching_files:
                missing_files.append(file_type)
        
        if missing_files:
            return False, f"Missing files: {', '.join(missing_files)}"
        
        return True, "All required files present"
    
    def clean_race_data(self, race_folder):
        """Clean and validate race data in a folder."""
        folder_path = os.path.join(self.data_dir, race_folder)
        
        print(f"🧹 Cleaning data in {race_folder}...")
        
        # Load and clean each file type
        files_cleaned = 0
        
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                file_path = os.path.join(folder_path, file)
                
                try:
                    df = pd.read_csv(file_path)
                    original_shape = df.shape
                    
                    # Basic cleaning
                    # Remove completely empty rows
                    df = df.dropna(how='all')
                    
                    # Clean column names
                    df.columns = df.columns.str.strip()
                    
                    # Handle common data issues
                    if 'Position' in df.columns:
                        # Clean position data
                        df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
                    
                    if 'Points' in df.columns:
                        # Clean points data
                        df['Points'] = pd.to_numeric(df['Points'], errors='coerce')
                    
                    # Save cleaned data
                    if df.shape != original_shape:
                        df.to_csv(file_path, index=False)
                        files_cleaned += 1
                        print(f"   ✓ Cleaned {file}: {original_shape} → {df.shape}")
                    
                except Exception as e:
                    print(f"   ❌ Error cleaning {file}: {e}")
        
        if files_cleaned > 0:
            print(f"✓ Cleaned {files_cleaned} files in {race_folder}")
        else:
            print(f"✓ No cleaning needed for {race_folder}")
        
        return files_cleaned > 0
    
    def create_data_summary(self):
        """Create a summary of all available data."""
        summary = {
            "last_updated": datetime.now(),
            "season": self.season,
            "available_races": [],
            "total_samples": 0,
            "data_quality": {}
        }
        
        # Find all GP folders
        gp_folders = [d for d in os.listdir(self.data_dir) 
                     if os.path.isdir(os.path.join(self.data_dir, d)) and "GP" in d]
        
        for folder in sorted(gp_folders):
            is_valid, message = self.validate_race_data(folder)
            
            race_info = {
                "folder": folder,
                "valid": is_valid,
                "message": message,
                "files": []
            }
            
            if is_valid:
                folder_path = os.path.join(self.data_dir, folder)
                for file in os.listdir(folder_path):
                    if file.endswith('.csv'):
                        file_path = os.path.join(folder_path, file)
                        try:
                            df = pd.read_csv(file_path)
                            race_info["files"].append({
                                "name": file,
                                "rows": len(df),
                                "columns": len(df.columns)
                            })
                            summary["total_samples"] += len(df)
                        except:
                            pass
            
            summary["available_races"].append(race_info)
        
        # Save summary
        with open("data_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📊 Data Summary:")
        print(f"   Available races: {len(gp_folders)}")
        print(f"   Valid races: {sum(1 for r in summary['available_races'] if r['valid'])}")
        print(f"   Total samples: {summary['total_samples']}")
        
        return summary
    
    def prepare_next_race_template(self):
        """Prepare template folder for next race."""
        next_race = self.get_next_race()
        if not next_race:
            print("No upcoming races found")
            return
        
        race_name = next_race["name"]
        folder_name = race_name.replace(" ", " ")
        folder_path = os.path.join(self.data_dir, folder_name)
        
        if os.path.exists(folder_path):
            print(f"Folder {folder_name} already exists")
            return
        
        # Create folder
        os.makedirs(folder_path, exist_ok=True)
        
        # Create template README
        readme_content = f"""# {race_name} Data
Date: {next_race["date"]}
Location: {next_race["location"]}
Round: {next_race["round"]}

## Required Files:
- qualifying_results_{self.season}_round_{next_race["round"]}.csv
- race_results_{self.season}_round_{next_race["round"]}.csv
- driver_standings_{self.season}_round_{next_race["round"]}.csv
- lap_times_{self.season}_round_{next_race["round"]}.csv
- pit_stops_{self.season}_round_{next_race["round"]}.csv

## Data Sources:
Add your data source information here.

## Notes:
Add any race-specific notes here.
"""
        
        with open(os.path.join(folder_path, "README.md"), "w") as f:
            f.write(readme_content)
        
        print(f"✓ Created template folder: {folder_name}")
        print(f"📅 Next race: {race_name} on {next_race['date']}")
        
        return folder_path

def setup_automatic_updates():
    """Setup automatic data checking and model updates."""
    
    def daily_check():
        """Daily check for new data and model updates."""
        print(f"\n🕐 Daily check at {datetime.now()}")
        
        # Initialize managers
        data_manager = F1DataManager()
        
        # Check for recent races
        recent_races = data_manager.get_recent_races(days=3)
        
        if recent_races:
            print(f"📅 Found {len(recent_races)} recent races")
            
            # Check if we have data for these races
            from update_model import F1ModelUpdater
            updater = F1ModelUpdater()
            
            # Try auto-update
            updater.auto_update()
        else:
            print("✓ No recent races to process")
        
        # Create data summary
        data_manager.create_data_summary()
        
        print("✓ Daily check completed")
    
    def weekly_cleanup():
        """Weekly data cleaning and maintenance."""
        print(f"\n🧹 Weekly cleanup at {datetime.now()}")
        
        data_manager = F1DataManager()
        
        # Find all GP folders and clean them
        gp_folders = [d for d in os.listdir(data_manager.data_dir) 
                     if os.path.isdir(os.path.join(data_manager.data_dir, d)) and "GP" in d]
        
        for folder in gp_folders:
            data_manager.clean_race_data(folder)
        
        print("✓ Weekly cleanup completed")
    
    # Schedule tasks
    schedule.every().day.at("09:00").do(daily_check)
    schedule.every().sunday.at("02:00").do(weekly_cleanup)
    
    print("⏰ Scheduled automatic updates:")
    print("   - Daily check: 9:00 AM")
    print("   - Weekly cleanup: Sunday 2:00 AM")
    print("   Press Ctrl+C to stop")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(3600)  # Check every hour
    except KeyboardInterrupt:
        print("\n👋 Stopping automatic updates")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='F1 Data Manager')
    parser.add_argument('--summary', action='store_true', help='Create data summary')
    parser.add_argument('--clean', type=str, help='Clean specific race folder')
    parser.add_argument('--prepare-next', action='store_true', help='Prepare next race template')
    parser.add_argument('--schedule', action='store_true', help='Run automatic scheduled updates')
    
    args = parser.parse_args()
    
    manager = F1DataManager()
    
    if args.summary:
        manager.create_data_summary()
    elif args.clean:
        manager.clean_race_data(args.clean)
    elif args.prepare_next:
        manager.prepare_next_race_template()
    elif args.schedule:
        setup_automatic_updates()
    else:
        print("F1 Data Manager - specify an action with --help")
