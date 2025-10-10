# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CredScope is a machine learning system for predicting credit default risk using the Home Credit Default Risk dataset from Kaggle. The project implements a phased approach with comprehensive feature engineering from 7 different data tables, achieving 0.7885 AUC through ensemble methods.

**Current Status**: Phase 2 Complete (522 features, 0.7885 AUC)

## Commands

### Data Setup
```bash
# Data files go in data/raw/ (not tracked in git)
# Required files: application_train.csv, application_test.csv, bureau.csv,
# bureau_balance.csv, previous_application.csv, installments_payments.csv,
# credit_card_balance.csv, POS_CASH_balance.csv
```

### Training Commands
```bash
# Phase 1: Baseline model (120 features, 0.7385 AUC)
python scripts/train_baseline.py

# Phase 2: Complete feature engineering (522 features, 0.7885 AUC)
python scripts/train_phase2_complete.py

# Phase 3: Hyperparameter optimization with Optuna
python scripts/train_phase3_optimisation.py

# Use cached features to skip feature engineering step
# Cached features are saved in data/features/engineered_features.pkl or engineered_features_COMPLETE.pkl
```

### MLflow Experiment Tracking
```bash
# View experiment results in browser
mlflow ui --host 0.0.0.0

# Then open http://localhost:5000
```

### Testing
```bash
# Run tests (when implemented)
pytest tests/

# Code formatting
black src/ scripts/

# Linting
flake8 src/ scripts/
```

## Architecture

### Data Pipeline Architecture

**Multi-Table Feature Engineering Flow**:
1. **DataLoader** (src/credscope/data/loader.py): Memory-efficient CSV loading with dtype optimization
2. **Feature Engineering Modules** (src/credscope/features/): Each table has dedicated feature engineering
3. **Aggregation Pattern**: All auxiliary tables aggregate to SK_ID_CURR (client ID) via left joins
4. **Caching System**: Engineered features cached as .pkl files to avoid recomputation

### Feature Engineering Architecture

The project uses **7 data tables** with specialized feature engineering for each:

**Core Application Data** (application_train/test.csv):
- Geographic features (src/credscope/features/geographic.py): Regional default rates, city patterns
- Document features (src/credscope/features/documents.py): Document submission patterns (22 document types)

**Bureau Credit History** (bureau.csv + bureau_balance.csv):
- src/credscope/features/bureau.py
- Aggregations: credit amounts, duration, overdue statistics
- Derived features: credit utilization, debt ratios, payment consistency
- Bureau balance features: Monthly payment status (DPD levels 0-5), trends, delinquency rates

**Previous Applications** (previous_application.csv):
- src/credscope/features/bureau.py (PreviousApplicationFeatureEngineer class)
- Approval/refusal/cancellation rates, amount patterns, interest rates
- Product and channel diversity metrics

**Payment Behavior Tables** (NEW in Phase 2, high predictive value):
- **Installments** (src/credscope/features/installments.py): Payment timeliness, late payment patterns (~50 features)
- **Credit Cards** (src/credscope/features/creditcard.py): Balance utilization, withdrawal patterns (~60 features)
- **POS/Cash** (src/credscope/features/pos_cash.py): Short-term credit behavior (~40 features)

**Key Insight**: 2 of top 10 features come from the Installments table (INST_RECENT_PAYMENT_RATIO, INST_RECENT_PAID_LATE)

### Model Architecture

**Ensemble Strategy**:
- LightGBM and XGBoost trained independently
- Early stopping with validation sets (stopping_rounds=100)
- Final predictions: Weighted average (typically 50/50 or optimized weights)

**Phase 3 Enhancement** (train_phase3_optimisation.py):
- Optuna hyperparameter optimization (150 trials per model, 3-fold CV)
- Feature selection using importance from quick LightGBM (top 200 features)
- Probability calibration with CalibratedClassifierCV (isotonic method)
- Multiple ensemble strategies: voting, blending, simple averaging

### Configuration System

- **config.yaml**: Central configuration for paths, model settings, MLflow tracking
- **src/credscope/utils/config.py**: load_config() function, path setup utilities
- All scripts use config-driven paths (data paths, MLflow URI, train/test split ratios)

## Key Technical Patterns

### Memory Optimization
The DataLoader.reduce_memory_usage() method optimizes dtypes (float64→float32/float16, int64→int8/16/32) to handle large datasets. This is critical because some tables have millions of rows (installments: 13.6M, POS_CASH: 10M).

### Feature Naming Convention
All engineered features use prefixes to indicate source table:
- `GEO_*`: Geographic features
- `DOC_*`: Document features
- `BUREAU_*`: Bureau credit history
- `BB_*`: Bureau balance (monthly payment status)
- `PREV_*`: Previous applications
- `INST_*`: Installments payments
- `CC_*`: Credit card balance
- `POS_*`: POS/Cash balance

This makes feature importance analysis easier and helps identify which data sources contribute most to predictions.

### Train/Test Column Alignment
Phase 2 scripts ensure train and test sets have identical columns (excluding TARGET):
```python
train_cols = set(train_df.columns) - {'TARGET'}
test_cols = set(test_df.columns)
common_cols = sorted(list(train_cols.intersection(test_cols)))
```

### Missing Value Strategy
- Numeric features: Fill with median from training set
- Categorical features: Fill with 'missing' string, then LabelEncode
- Handle inf/-inf values: Replace with np.nan before filling

## Important Implementation Notes

### Type Handling for MLflow
When logging metrics to MLflow, always convert numpy scalars to Python float:
```python
mlflow.log_metric("auc", float(auc_score))  # Not just auc_score
```

### LightGBM predict_proba Indexing
Add type ignore comments when indexing predict_proba results:
```python
y_pred = model.predict_proba(X)[:, 1]  # type: ignore
```

### Optuna Objective Return Type
Ensure Optuna objective functions return Python float, not numpy scalar:
```python
return float(np.mean(scores))  # Not just np.mean(scores)
```

### Feature Caching
The train_phase2_complete.py script caches all 522 features to data/features/engineered_features_COMPLETE.pkl. Use `use_cached=True` parameter to skip the 15-25 minute feature engineering step:
```python
train_df, test_df, feature_names = load_and_engineer_features_complete(
    config, use_cached=True  # Loads from cache
)
```

## Current Performance Baseline

| Phase | Features | Models | AUC | Status |
|-------|----------|--------|-----|--------|
| Phase 1 | 120 | Logistic Regression | 0.7385 | Baseline |
| Phase 2 (Partial) | 303 | LightGBM + XGBoost | 0.7794 | Superseded |
| Phase 2 (Complete) | 522 | LightGBM + XGBoost Ensemble | 0.7885 | Current |
| Phase 3 | 200 (selected) | Optimized + Calibrated + Stacking | TBD | Next |

**Target**: 0.80+ AUC (currently -0.0115 gap)

## Development Workflow

### When Adding New Features
1. Create feature engineering function in appropriate module (src/credscope/features/)
2. Follow naming convention (e.g., `CC_*` for credit card features)
3. Add integration to train_phase2_complete.py pipeline
4. Clear feature cache or use `use_cached=False` to regenerate
5. Compare feature importance in models/feature_importance_complete.csv

### When Optimizing Models
1. Use Phase 3 script with Optuna for hyperparameter tuning
2. Start with narrower search ranges (see QUICK_FIXES_IMPLEMENTATION.md)
3. Monitor with MLflow UI to track trial results
4. Save best parameters to models/phase3_enhanced_metadata.pkl

### When Debugging Training Issues
- Check MLflow UI for run history and metrics
- Feature importance CSV shows which features contribute most
- Validation AUC gap from train AUC indicates overfitting
- Memory issues: Use reduce_memory_usage() or reduce nrows parameter

## Project Structure Details

```
credscope/
├── config.yaml              # Central configuration
├── data/
│   ├── raw/                 # CSV files from Kaggle (not in git)
│   ├── processed/           # Intermediate processed data
│   └── features/            # Cached engineered features (.pkl)
├── models/                  # Trained models and metadata (.pkl, .txt, .json)
├── mlruns/                  # MLflow experiment tracking data
├── scripts/
│   ├── train_baseline.py               # Phase 1 baseline
│   ├── train_phase2_complete.py        # Phase 2 with all 7 tables (CURRENT)
│   └── train_phase3_optimisation.py    # Phase 3 optimization (NEXT)
├── src/credscope/
│   ├── data/
│   │   └── loader.py        # DataLoader class with memory optimization
│   ├── features/            # Feature engineering modules (one per table)
│   │   ├── geographic.py
│   │   ├── documents.py
│   │   ├── bureau.py        # Bureau + Previous Applications
│   │   ├── installments.py
│   │   ├── creditcard.py
│   │   └── pos_cash.py
│   ├── models/
│   │   ├── baseline.py      # Baseline logistic regression
│   │   └── tuner.py         # Hyperparameter tuning utilities
│   ├── evaluation/
│   │   └── explainer.py     # SHAP and model interpretation
│   └── utils/
│       └── config.py        # Configuration loading
└── notebooks/               # Jupyter notebooks for exploration
```

## Next Steps (Phase 3 Path)

Based on PHASE2_COMPLETE_RESULTS_AND_NEXT_STEPS.md, the recommended path to reach 0.80+ AUC:

1. **Apply Quick Fixes** (30 min, +0.005-0.008 AUC): Narrow hyperparameter ranges, remove feature selection, add class_weight='balanced'
2. **Hyperparameter Optimization** (2-3 hours, +0.005-0.010 AUC): Run Optuna with 150 trials on all 522 features
3. **Advanced Features** (1 hour, +0.003-0.008 AUC): Add 20 high-value interaction features (EXT_SOURCE combinations, payment ratios)
4. **Enhanced Ensemble** (30 min, +0.002-0.005 AUC): Add CatBoost, implement stacking with LogisticRegression meta-learner

**Expected Final Result**: 0.803-0.815 AUC
