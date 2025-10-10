# Quick Fixes Implementation Guide

## 🚀 Immediate Actions (Do These Now)

### **Fix 1: Remove Feature Selection (30 minutes)**

**File**: `scripts/train_phase3_optimisation.py`

**Change lines 405-410** from:
```python
top_features, X_train_selected, X_val_selected = select_top_features(
    X_train, y_train, X_val, n_features=200
)

mlflow.log_param("n_features_selected", len(top_features))
```

**To**:
```python
# Use all features - no selection (tree models handle dimensionality well)
X_train_selected = X_train
X_val_selected = X_val
top_features = X_train.columns.tolist()

mlflow.log_param("n_features_selected", len(top_features))
mlflow.log_param("feature_selection", "none (using all features)")
```

**Expected gain**: +0.003-0.006 AUC

---

### **Fix 2: Narrow Hyperparameter Ranges (1 hour)**

**File**: `scripts/train_phase3_optimisation.py`

**Lines 105-132** - LightGBM parameters - Change to:
```python
params = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'random_state': 42,
    'force_col_wise': True,
    'class_weight': 'balanced',  # ADD THIS for imbalanced data
    
    # Narrowed tree structure (focus around proven values)
    'num_leaves': trial.suggest_int('num_leaves', 50, 100),  # Was: 10-200
    'max_depth': trial.suggest_int('max_depth', 5, 9),       # Was: 2-15
    'min_child_samples': trial.suggest_int('min_child_samples', 20, 80),  # Was: 5-200
    'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 10.0, log=True),  # Was: 1e-4-100
    
    # Better learning rate range
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),  # Was: 0.001-0.3
    'n_estimators': trial.suggest_int('n_estimators', 500, 2000),  # Was: 200-3000
    
    # Reasonable sampling (less aggressive)
    'feature_fraction': trial.suggest_float('feature_fraction', 0.65, 0.95),  # Was: 0.3-1.0
    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.65, 0.95),  # Was: 0.3-1.0
    'bagging_freq': trial.suggest_int('bagging_freq', 3, 7),  # Was: 1-10
    
    # Focused regularization
    'lambda_l1': trial.suggest_float('lambda_l1', 0.001, 5.0, log=True),  # Was: 1e-8-100
    'lambda_l2': trial.suggest_float('lambda_l2', 0.001, 5.0, log=True),  # Was: 1e-8-100
    'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0, 5),  # Was: 0-20
    
    # Reasonable additional parameters
    'path_smooth': trial.suggest_float('path_smooth', 0, 0.5),  # Was: 0-1
    'max_bin': trial.suggest_int('max_bin', 200, 300),  # Was: 100-500
}
```

**Lines 183-212** - XGBoost parameters - Change to:
```python
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'random_state': 42,
    'verbosity': 0,
    
    # Narrowed tree structure
    'max_depth': trial.suggest_int('max_depth', 4, 8),  # Was: 2-12
    'min_child_weight': trial.suggest_int('min_child_weight', 10, 100),  # Was: 1-500
    'gamma': trial.suggest_float('gamma', 0.001, 2.0, log=True),  # Was: 1e-8-10
    
    # Better learning range
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),  # Was: 0.001-0.3
    'n_estimators': trial.suggest_int('n_estimators', 500, 2000),  # Was: 200-3000
    
    # Reasonable sampling
    'subsample': trial.suggest_float('subsample', 0.65, 0.95),  # Was: 0.3-1.0
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 0.95),  # Was: 0.3-1.0
    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.65, 0.95),  # Was: 0.3-1.0
    'colsample_bynode': trial.suggest_float('colsample_bynode', 0.65, 0.95),  # Was: 0.3-1.0
    
    # Focused regularization
    'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 5.0, log=True),  # Was: 1e-8-100
    'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 5.0, log=True),  # Was: 1e-8-100
    
    # Reasonable additional
    'max_bin': trial.suggest_int('max_bin', 200, 300),  # Was: 100-500
    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 8, 15),  # Was: 1-20 (narrow around 8% default rate)
}
```

**Lines 167-171** - Increase trials and improve sampler:
```python
study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(
        seed=42, 
        n_startup_trials=30,  # Was: 20 - more random exploration
        n_ei_candidates=24,   # ADD THIS - more candidates
        multivariate=True     # ADD THIS - consider parameter interactions
    )
)

study.optimize(objective, n_trials=200, show_progress_bar=True)  # Was: 150
```

**Expected gain**: +0.004-0.008 AUC

---

### **Fix 3: Change from 3-fold to 5-fold CV (15 minutes)**

**Lines 136 and 214** - Change CV folds:
```python
# FROM:
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# TO:
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # More stable estimates
```

**Expected gain**: More stable validation, +0.001-0.002 AUC

---

## 📝 Summary of Changes

**Total time**: ~2 hours  
**Expected combined gain**: +0.008 to +0.016 AUC  
**New expected AUC**: 0.7874 to 0.7954

### **What These Fixes Do:**

1. **No Feature Selection**: 
   - Keeps all 303 engineered features
   - Preserves feature interactions
   - Let models naturally select important features

2. **Narrowed Hyperparameters**:
   - Focus search around proven good values
   - Eliminate obviously bad parameter combinations
   - More efficient use of 200 trials
   - Better Bayesian optimization with multivariate sampler

3. **5-Fold CV**:
   - More stable validation estimates
   - Better generalization assessment
   - Only ~20% slower than 3-fold

4. **Class Weights**:
   - Handles imbalanced data (92% repaid, 8% default)
   - Prevents model from just predicting "no default" for everyone
   - Forces model to learn minority class patterns

---

## 🔄 How to Apply

### **Option 1: Stop Current Run and Restart**
```bash
# Stop the current training (Ctrl+C)
# Edit the file with changes above
# Restart:
python scripts/train_phase3_optimisation.py
```

### **Option 2: Let Current Run Finish, Then Run Improved Version**
```bash
# Let current 150 trials complete
# Save as train_phase3_improved.py
# Run new version:
python scripts/train_phase3_improved.py
```

**Recommended**: Option 2 - let current run finish for comparison

---

## 📊 After These Fixes

Once you've implemented these quick fixes and re-run Phase 3, we should see:

**Before**: 0.7794 (Phase 2) → 0.7751 (Phase 3, declining)  
**After**: 0.7794 (Phase 2) → **0.7870-0.7950** (Phase 3, improving!)

Then we can tackle the big wins:
- Add missing data tables (+0.010-0.020 AUC)
- Add interaction features (+0.005-0.010 AUC)
- Improve ensemble (+0.003-0.007 AUC)

**Final target: 0.80-0.82 AUC ✨**

---

## 💡 Quick Test Before Full Run

To verify changes work without waiting 1 hour:
```python
# In main(), add after line 380:
if __name__ == "__main__":
    # Quick test with 10 trials
    import sys
    if '--test' in sys.argv:
        n_trials = 10
    else:
        n_trials = 200
```

Then test:
```bash
python scripts/train_phase3_optimisation.py --test
```

If no errors, run full:
```bash
python scripts/train_phase3_optimisation.py
```
