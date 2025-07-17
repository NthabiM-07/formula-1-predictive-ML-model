"""
F1 Project Cleanup Script
========================

This script identifies and removes unnecessary files from your F1 prediction project.
It keeps only essential files and removes duplicates, old backups, and temporary files.
"""

import os
import shutil
import glob
from pathlib import Path
import json

class F1ProjectCleaner:
    def __init__(self, data_dir="."):
        """Initialize the project cleaner."""
        self.data_dir = data_dir
        self.essential_files = {
            # Core model files
            'f1_prediction_model.py',
            'f1_prediction_model.joblib',
            
            # Main scripts
            'update_model.py',
            'test_model.py',
            'quick_test.py',
            'live_test.py',
            'historical_integrator.py',
            'data_manager.py',
            
            # Documentation
            'README.md',
            'UPDATE_WORKFLOW.md',
            
            # Configuration
            'model_config.json',
            'race_calendar_2025.json',
            'data_summary.json'
        }
        
        self.essential_directories = {
            'models',           # Model backups
            'Austria GP',       # Current race data
            'historical_data'   # Historical data (if exists)
        }
        
        self.removable_patterns = [
            '*.pyc',
            '*.pyo',
            '__pycache__',
            '.pytest_cache',
            '*.tmp',
            '*.temp',
            'Untitled*.ipynb',
            '*.log'
        ]
        
        self.duplicate_candidates = [
            'predict*.py',      # Multiple prediction scripts
            'feature_*.py',     # Feature engineering files
            'test_*.png',       # Test output images
            'actual_*.png',     # Visualization outputs
            'race_prediction_*.csv'  # Old prediction files
        ]

    def analyze_project(self):
        """Analyze the project structure and identify files for cleanup."""
        print("🔍 Analyzing F1 project structure...")
        
        analysis = {
            'total_files': 0,
            'total_size_mb': 0,
            'essential_files': [],
            'removable_files': [],
            'duplicate_files': [],
            'large_files': [],
            'old_files': [],
            'directories': []
        }
        
        # Scan all files
        for root, dirs, files in os.walk(self.data_dir):
            # Skip certain directories
            if any(skip_dir in root for skip_dir in ['.git', '.venv', '__pycache__']):
                continue
                
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, self.data_dir)
                
                try:
                    file_size = os.path.getsize(file_path)
                    file_size_mb = file_size / (1024 * 1024)
                    
                    analysis['total_files'] += 1
                    analysis['total_size_mb'] += file_size_mb
                    
                    file_info = {
                        'path': relative_path,
                        'size_mb': file_size_mb,
                        'modified': os.path.getmtime(file_path)
                    }
                    
                    # Categorize files
                    if file in self.essential_files:
                        analysis['essential_files'].append(file_info)
                    elif any(pattern in file.lower() for pattern in ['cache', 'temp', 'tmp', '.pyc']):
                        analysis['removable_files'].append(file_info)
                    elif file_size_mb > 10:  # Large files
                        analysis['large_files'].append(file_info)
                    elif self.is_duplicate_candidate(file):
                        analysis['duplicate_files'].append(file_info)
                    
                except (OSError, IOError):
                    pass
        
        # Analyze directories
        for item in os.listdir(self.data_dir):
            item_path = os.path.join(self.data_dir, item)
            if os.path.isdir(item_path):
                dir_size = self.get_directory_size(item_path)
                analysis['directories'].append({
                    'name': item,
                    'size_mb': dir_size / (1024 * 1024),
                    'essential': item in self.essential_directories
                })
        
        return analysis
    
    def is_duplicate_candidate(self, filename):
        """Check if file might be a duplicate based on patterns."""
        for pattern in self.duplicate_candidates:
            if self.matches_pattern(filename, pattern):
                return True
        return False
    
    def matches_pattern(self, filename, pattern):
        """Check if filename matches a glob pattern."""
        import fnmatch
        return fnmatch.fnmatch(filename.lower(), pattern.lower())
    
    def get_directory_size(self, directory):
        """Calculate total size of directory."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except (OSError, IOError):
                        pass
        except (OSError, IOError):
            pass
        return total_size
    
    def print_analysis(self, analysis):
        """Print analysis results."""
        print(f"\n📊 Project Analysis Results:")
        print(f"=" * 50)
        print(f"Total Files: {analysis['total_files']}")
        print(f"Total Size: {analysis['total_size_mb']:.1f} MB")
        
        print(f"\n✅ Essential Files ({len(analysis['essential_files'])}):")
        for file_info in analysis['essential_files'][:10]:  # Show first 10
            print(f"  - {file_info['path']} ({file_info['size_mb']:.1f} MB)")
        
        print(f"\n🗑️ Removable Files ({len(analysis['removable_files'])}):")
        removable_size = sum(f['size_mb'] for f in analysis['removable_files'])
        print(f"  Total size: {removable_size:.1f} MB")
        for file_info in analysis['removable_files'][:10]:
            print(f"  - {file_info['path']} ({file_info['size_mb']:.1f} MB)")
        
        print(f"\n🔄 Potential Duplicates ({len(analysis['duplicate_files'])}):")
        duplicate_size = sum(f['size_mb'] for f in analysis['duplicate_files'])
        print(f"  Total size: {duplicate_size:.1f} MB")
        for file_info in analysis['duplicate_files'][:10]:
            print(f"  - {file_info['path']} ({file_info['size_mb']:.1f} MB)")
        
        print(f"\n📁 Directories:")
        for dir_info in sorted(analysis['directories'], key=lambda x: x['size_mb'], reverse=True):
            status = "✅ Essential" if dir_info['essential'] else "❓ Review"
            print(f"  - {dir_info['name']}: {dir_info['size_mb']:.1f} MB ({status})")
        
        # Calculate potential savings
        potential_savings = removable_size + duplicate_size
        print(f"\n💾 Potential Space Savings: {potential_savings:.1f} MB")
    
    def suggest_cleanup_actions(self, analysis):
        """Suggest specific cleanup actions."""
        print(f"\n🧹 Cleanup Recommendations:")
        print(f"=" * 50)
        
        actions = []
        
        # Removable files
        if analysis['removable_files']:
            actions.append({
                'action': 'Remove temporary/cache files',
                'files': analysis['removable_files'],
                'safe': True,
                'savings_mb': sum(f['size_mb'] for f in analysis['removable_files'])
            })
        
        # Old prediction files
        old_predictions = [f for f in analysis['duplicate_files'] 
                          if 'prediction' in f['path'].lower()]
        if old_predictions:
            actions.append({
                'action': 'Remove old prediction CSV files',
                'files': old_predictions,
                'safe': True,
                'savings_mb': sum(f['size_mb'] for f in old_predictions)
            })
        
        # Duplicate visualization files
        old_plots = [f for f in analysis['duplicate_files'] 
                    if any(ext in f['path'].lower() for ext in ['.png', '.jpg', '.pdf'])]
        if old_plots:
            actions.append({
                'action': 'Remove old visualization files',
                'files': old_plots,
                'safe': True,
                'savings_mb': sum(f['size_mb'] for f in old_plots)
            })
        
        # Archive files
        archives = [f for f in analysis.get('large_files', []) 
                   if any(ext in f['path'].lower() for ext in ['.zip', '.tar', '.gz'])]
        if archives:
            actions.append({
                'action': 'Remove archive files (extract first if needed)',
                'files': archives,
                'safe': False,
                'savings_mb': sum(f['size_mb'] for f in archives)
            })
        
        for i, action in enumerate(actions, 1):
            safety = "✅ Safe" if action['safe'] else "⚠️ Review first"
            print(f"\n{i}. {action['action']} ({safety})")
            print(f"   Files: {len(action['files'])}")
            print(f"   Savings: {action['savings_mb']:.1f} MB")
            
            # Show first few files
            for file_info in action['files'][:3]:
                print(f"   - {file_info['path']}")
            if len(action['files']) > 3:
                print(f"   ... and {len(action['files']) - 3} more")
        
        return actions
    
    def execute_cleanup(self, actions, auto_confirm=False):
        """Execute cleanup actions."""
        print(f"\n🧹 Executing Cleanup...")
        print(f"=" * 30)
        
        total_removed = 0
        total_saved_mb = 0
        
        for action in actions:
            if not action['safe'] and not auto_confirm:
                response = input(f"\n⚠️ {action['action']} - This requires review. Continue? (y/N): ")
                if response.lower() != 'y':
                    print(f"   Skipped: {action['action']}")
                    continue
            
            print(f"\n🗑️ {action['action']}...")
            
            for file_info in action['files']:
                file_path = os.path.join(self.data_dir, file_info['path'])
                
                try:
                    if os.path.exists(file_path):
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                        
                        total_removed += 1
                        total_saved_mb += file_info['size_mb']
                        print(f"   ✓ Removed: {file_info['path']}")
                    
                except Exception as e:
                    print(f"   ❌ Error removing {file_info['path']}: {e}")
        
        print(f"\n✅ Cleanup Complete!")
        print(f"   Files removed: {total_removed}")
        print(f"   Space saved: {total_saved_mb:.1f} MB")
    
    def clean_specific_patterns(self):
        """Clean files matching specific patterns."""
        print("🧹 Cleaning specific file patterns...")
        
        removed_count = 0
        
        for pattern in self.removable_patterns:
            matches = glob.glob(os.path.join(self.data_dir, '**', pattern), recursive=True)
            
            for match in matches:
                try:
                    if os.path.isfile(match):
                        os.remove(match)
                        print(f"   ✓ Removed: {os.path.relpath(match, self.data_dir)}")
                        removed_count += 1
                    elif os.path.isdir(match):
                        shutil.rmtree(match)
                        print(f"   ✓ Removed directory: {os.path.relpath(match, self.data_dir)}")
                        removed_count += 1
                except Exception as e:
                    print(f"   ❌ Error removing {match}: {e}")
        
        print(f"✅ Removed {removed_count} files/directories")

def main():
    """Main cleanup function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='F1 Project Cleanup Tool')
    parser.add_argument('--analyze', action='store_true', help='Analyze project only')
    parser.add_argument('--clean-safe', action='store_true', help='Clean only safe files')
    parser.add_argument('--clean-all', action='store_true', help='Clean all recommended files')
    parser.add_argument('--clean-patterns', action='store_true', help='Clean specific patterns only')
    
    args = parser.parse_args()
    
    cleaner = F1ProjectCleaner()
    
    if args.analyze:
        analysis = cleaner.analyze_project()
        cleaner.print_analysis(analysis)
        cleaner.suggest_cleanup_actions(analysis)
        
    elif args.clean_patterns:
        cleaner.clean_specific_patterns()
        
    elif args.clean_safe or args.clean_all:
        analysis = cleaner.analyze_project()
        actions = cleaner.suggest_cleanup_actions(analysis)
        
        # Filter to safe actions only
        if args.clean_safe:
            actions = [a for a in actions if a['safe']]
        
        if actions:
            cleaner.execute_cleanup(actions, auto_confirm=args.clean_all)
        else:
            print("✅ No cleanup actions needed!")
    
    else:
        print("F1 Project Cleanup Tool")
        print("=" * 30)
        print("Use --analyze to see what can be cleaned")
        print("Use --clean-safe to clean safe files only")
        print("Use --clean-patterns to clean cache/temp files")

if __name__ == "__main__":
    main()
