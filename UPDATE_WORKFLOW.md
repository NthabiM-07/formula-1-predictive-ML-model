# F1 Model Update Workflow

This document explains how to keep your F1 prediction model updated throughout the season.

## 📅 Update Strategy

### **After Each Race (Recommended)**
```bash
# Automatic detection and update
python update_model.py --auto-update

# Or manually specify the race
python update_model.py --race "Hungary GP" --round 13
```

### **Weekly Maintenance**
```bash
# Clean and validate all data
python data_manager.py --summary

# Clean specific race data if needed
python data_manager.py --clean "Austria GP"
```

### **Mid-Season Full Retrain**
```bash
# Every 5-6 races, do a full retrain
python update_model.py --full-retrain
```

## 🔄 How Updates Work

### **1. Incremental Updates (Default)**
- Adds new race data to existing dataset
- Retrains model with expanded data
- Only keeps new model if performance improves
- Fast and efficient for regular updates

### **2. Full Retrain**
- Retrains from scratch with all available data
- More thorough but takes longer
- Recommended every few races or mid-season

### **3. Automatic Model Management**
- Backs up previous models before updating
- Tracks performance history
- Version control for models
- Rollback capability if updates fail

## 📊 Performance Tracking

The system tracks:
- **R² Score**: Model accuracy
- **RMSE**: Prediction error in positions
- **Data Size**: Number of training samples
- **Version History**: All model versions

### **Update Thresholds**
- Model updates only if R² improves by ≥0.05 (5%)
- Or if RMSE improves by ≥1.0 position
- Prevents degradation from noisy data

## 🗂️ File Organization

```
F1 data/
├── models/                          # Model backups
│   ├── f1_model_v1.0.0_20250712.joblib
│   └── f1_model_v1.0.1_20250720.joblib
├── f1_prediction_model.joblib       # Current active model
├── model_config.json               # Model metadata & history
├── data_summary.json               # Data quality report
└── Race Folders/
    ├── Austria GP/
    ├── Hungary GP/
    └── Belgium GP/
```

## 🚀 Quick Start Workflow

### **1. Set Up Automatic Updates**
```bash
# Check current status
python update_model.py --status

# Prepare for next race
python data_manager.py --prepare-next
```

### **2. After Each Race Weekend**
```bash
# 1. Add new race data to appropriate folder
# 2. Run automatic update
python update_model.py --auto-update

# 3. Test the updated model
python test_model.py
```

### **3. Validate Performance**
```bash
# Test with latest race data
python live_test.py

# Check prediction accuracy
python quick_test.py
```

## 📈 Expected Performance Evolution

### **Early Season (Races 1-5)**
- **R² Score**: 0.10-0.25
- **Dataset**: 100-200 samples
- **Strategy**: Update after every race

### **Mid Season (Races 6-15)**
- **R² Score**: 0.25-0.45
- **Dataset**: 300-600 samples
- **Strategy**: Update every 2-3 races + full retrain

### **Late Season (Races 16-24)**
- **R² Score**: 0.40-0.60
- **Dataset**: 600+ samples
- **Strategy**: Weekly updates + validation

## ⚠️ Best Practices

### **Data Quality**
1. **Always validate new data** before updating
2. **Clean data** regularly to remove inconsistencies
3. **Monitor feature distributions** for anomalies

### **Update Timing**
1. **Update soon after race** while data is fresh
2. **Don't update during race weekend** (use for predictions)
3. **Schedule updates** during quiet periods

### **Performance Monitoring**
1. **Track prediction accuracy** on recent races
2. **Monitor feature importance** changes
3. **Validate on out-of-sample data**

### **Backup Strategy**
1. **Always backup** before major updates
2. **Keep multiple versions** for rollback
3. **Test thoroughly** before deploying

## 🔧 Troubleshooting

### **Model Performance Drops**
```bash
# Check data quality
python data_manager.py --summary

# Restore previous model
cp models/f1_model_v1.0.0_*.joblib f1_prediction_model.joblib

# Full retrain with cleaned data
python update_model.py --full-retrain
```

### **Update Fails**
```bash
# Check logs and fix data issues
python data_manager.py --clean "ProblemRace GP"

# Try manual update
python update_model.py --race "FixedRace GP" --round X
```

### **Poor Predictions**
```bash
# Validate recent performance
python live_test.py

# Check feature importance
python test_model.py

# Consider feature engineering improvements
python feature_engineering_suggestions.py
```

## 📞 Model Status Commands

```bash
# Current model information
python update_model.py --status

# Data summary and quality
python data_manager.py --summary

# Performance testing
python test_model.py

# Live prediction test
python live_test.py
```

## 🎯 Season Goals

### **Target Metrics by Season End**
- **R² Score**: >0.50 (50% variance explained)
- **RMSE**: <3.0 positions
- **Top 3 Accuracy**: >70%
- **Points Accuracy**: >60%

### **Data Requirements**
- **Minimum**: 400+ samples (20 races × 20 drivers)
- **Optimal**: 600+ samples with historical data
- **Features**: 8-15 meaningful features

This systematic approach ensures your model improves throughout the season while maintaining reliability and performance tracking.
