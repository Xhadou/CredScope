"""
Phase 3: Advanced Optimization & Feature Engineering
Target: 0.80+ AUC (from current 0.7885)

Key improvements:
1. Load cached 522 features from Phase 2
2. Create 20 high-value interaction features
3. Optimize hyperparameters with Optuna (150 trials/model)
4. Handle class imbalance properly
5. Create stacking ensemble
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import joblib
from datetime import datetime
import warnings
import gc
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
import optuna
from optuna.integration import LightGBMPruningCallback, XGBoostPruningCallback
import mlflow

# Try to import CatBoost (optional)
try:
    import catboost as cb
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
    print("✅ CatBoost available - will use 3-model ensemble")
except ImportError:
    HAS_CATBOOST = False
    print("⚠️ CatBoost not installed - will use 2-model ensemble")

# Our modules
from src.credscope.utils.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress Optuna logs during optimization
optuna.logging.set_verbosity(optuna.logging.WARNING)


def create_interaction_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Create 20 high-value interaction features based on feature importance
    
    Args:
        df: DataFrame with all features
        is_train: Whether this is training data (for logging)
        
    Returns:
        DataFrame with interaction features added
    """
    print(f"\n🔧 Creating interaction features...")
    
    interactions = pd.DataFrame(index=df.index)
    
    # 1. External source interactions (highest importance features)
    if 'EXT_SOURCE_2' in df.columns and 'EXT_SOURCE_3' in df.columns:
        interactions['INT_EXT_23_MULT'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
        interactions['INT_EXT_23_RATIO'] = df['EXT_SOURCE_2'] / (df['EXT_SOURCE_3'] + 0.0001)
    
    if 'EXT_SOURCE_1' in df.columns and 'EXT_SOURCE_2' in df.columns:
        interactions['INT_EXT_12_MULT'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
        interactions['INT_EXT_12_DIFF'] = df['EXT_SOURCE_1'] - df['EXT_SOURCE_2']
    
    # Average of all external sources
    ext_cols = [col for col in df.columns if col.startswith('EXT_SOURCE_')]
    if ext_cols:
        interactions['INT_EXT_AVG'] = df[ext_cols].mean(axis=1)
        interactions['INT_EXT_STD'] = df[ext_cols].std(axis=1)
        interactions['INT_EXT_MAX'] = df[ext_cols].max(axis=1)
        interactions['INT_EXT_MIN'] = df[ext_cols].min(axis=1)
    
    # 2. Age-credit interactions
    if 'DAYS_BIRTH' in df.columns and 'AMT_CREDIT' in df.columns:
        interactions['INT_AGE_CREDIT'] = (-df['DAYS_BIRTH'] / 365.25) * df['AMT_CREDIT']
        interactions['INT_AGE_ANNUITY'] = (-df['DAYS_BIRTH'] / 365.25) * df.get('AMT_ANNUITY', 0)
    
    # 3. Employment-income interactions
    if 'DAYS_EMPLOYED' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        interactions['INT_EMPLOY_INCOME'] = (-df['DAYS_EMPLOYED'] / 365.25) * df['AMT_INCOME_TOTAL']
        interactions['INT_INCOME_PER_YEAR_EMPLOYED'] = df['AMT_INCOME_TOTAL'] / ((-df['DAYS_EMPLOYED'] / 365.25) + 0.1)
    
    # 4. Payment behavior interactions (from installments)
    if 'INST_RECENT_INST_PAYMENT_RATIO' in df.columns:
        interactions['INT_PAYMENT_BUREAU'] = df.get('INST_RECENT_INST_PAYMENT_RATIO', 0) * df.get('BUREAU_DEBT_CREDIT_RATIO', 0)
        interactions['INT_PAYMENT_LATE'] = df.get('INST_RECENT_INST_PAYMENT_RATIO', 0) * df.get('INST_RECENT_INST_PAID_LATE', 0)
    
    # 5. Credit utilization interactions
    if 'CC_VERY_RECENT_CC_UTILIZATION' in df.columns and 'BUREAU_DEBT_CREDIT_RATIO' in df.columns:
        interactions['INT_UTIL_DEBT'] = df['CC_VERY_RECENT_CC_UTILIZATION'] * df['BUREAU_DEBT_CREDIT_RATIO']
    
    # 6. Document quality interactions
    if 'DOC_PER_CREDIT_UNIT' in df.columns:
        interactions['INT_DOC_CREDIT'] = df['DOC_PER_CREDIT_UNIT'] * df.get('AMT_CREDIT', 0)
        interactions['INT_DOC_INCOME'] = df['DOC_PER_CREDIT_UNIT'] * df.get('AMT_INCOME_TOTAL', 0)
    
    # 7. Gender-based interactions (often important in credit)
    if 'CODE_GENDER' in df.columns:
        interactions['INT_GENDER_INCOME'] = df['CODE_GENDER'] * df.get('AMT_INCOME_TOTAL', 0)
        interactions['INT_GENDER_CREDIT'] = df['CODE_GENDER'] * df.get('AMT_CREDIT', 0)
    
    # Handle infinities and NaNs
    interactions = interactions.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with median for each column
    for col in interactions.columns:
        median_val = interactions[col].median()
        if pd.isna(median_val):
            median_val = 0
        interactions[col] = interactions[col].fillna(median_val)
    
    # Add interactions to original dataframe
    result = pd.concat([df, interactions], axis=1)
    
    if is_train:
        logger.info(f"✅ Created {len(interactions.columns)} interaction features")
        logger.info(f"   Total features now: {len(result.columns)}")
    
    return result


def optimize_lightgbm(X_train, y_train, X_val, y_val, n_trials=150):
    """Optimize LightGBM hyperparameters using Optuna
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        n_trials: Number of optimization trials
        
    Returns:
        Best model and best AUC score
    """
    print(f"\n🔍 Optimizing LightGBM ({n_trials} trials)...")
    print("   This will take 30-45 minutes...")
    
    def objective(trial):
        # Focused hyperparameter ranges based on experience
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': 42,
            
            # Handle imbalanced data
            'is_unbalance': True,
            
            # Tree parameters
            'num_leaves': trial.suggest_int('num_leaves', 50, 100),
            'max_depth': trial.suggest_int('max_depth', 5, 9),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
            
            # Regularization
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.65, 0.95),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.65, 0.95),
            'bagging_freq': trial.suggest_int('bagging_freq', 3, 7),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.001, 5.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.001, 5.0, log=True),
        }
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train with early stopping
        pruning_callback = LightGBMPruningCallback(trial, "valid_0-auc")
        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[val_data],
            valid_names=['valid_0'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(0),  # Suppress output
                pruning_callback
            ]
        )
        
        # Get validation score
        val_pred = model.predict(X_val)
        val_auc = roc_auc_score(y_val, val_pred)  # type: ignore[arg-type]
        
        return float(val_auc)
    
    # Run optimization
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)  # type: ignore[arg-type]
    
    # Train final model with best params
    best_params = study.best_params
    best_params.update({
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'is_unbalance': True,
        'verbosity': -1,
        'random_state': 42
    })
    
    print(f"\n   Best LightGBM params found:")
    for key, value in sorted(study.best_params.items()):
        print(f"     {key}: {value}")
    
    # Train final model
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    final_model = lgb.train(
        best_params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(100)
        ]
    )
    
    val_pred = final_model.predict(X_val)
    val_auc = roc_auc_score(y_val, val_pred)  # type: ignore[arg-type]
    
    logger.info(f"✅ LightGBM optimized AUC: {val_auc:.4f}")
    
    return final_model, val_auc


def optimize_xgboost(X_train, y_train, X_val, y_val, n_trials=150):
    """Optimize XGBoost hyperparameters using Optuna
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        n_trials: Number of optimization trials
        
    Returns:
        Best model and best AUC score
    """
    print(f"\n🔍 Optimizing XGBoost ({n_trials} trials)...")
    print("   This will take 30-45 minutes...")
    
    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    
    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'random_state': 42,
            
            # Handle imbalanced data
            'scale_pos_weight': scale_pos_weight,
            
            # Tree parameters
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            
            # Regularization
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'subsample': trial.suggest_float('subsample', 0.65, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.001, 1.0, log=True),
        }
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train with early stopping
        pruning_callback = XGBoostPruningCallback(trial, "validation-auc")
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=[(dval, 'validation')],
            early_stopping_rounds=50,
            verbose_eval=False,
            callbacks=[pruning_callback]
        )
        
        # Get validation score
        val_pred = model.predict(dval)
        val_auc = roc_auc_score(y_val, val_pred)  # type: ignore[arg-type]
        
        return float(val_auc)
    
    # Run optimization
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)  # type: ignore[arg-type]
    
    # Train final model with best params
    best_params = study.best_params
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42
    })
    
    print(f"\n   Best XGBoost params found:")
    for key, value in sorted(study.best_params.items()):
        print(f"     {key}: {value}")
    
    # Train final model
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    final_model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=2000,
        evals=[(dtrain, 'train'), (dval, 'valid')],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    val_pred = final_model.predict(dval)
    val_auc = roc_auc_score(y_val, val_pred)
    
    logger.info(f"✅ XGBoost optimized AUC: {val_auc:.4f}")
    
    return final_model, val_auc


def optimize_catboost(X_train, y_train, X_val, y_val, n_trials=150):
    """Optimize CatBoost hyperparameters using Optuna (if available)
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        n_trials: Number of optimization trials
        
    Returns:
        Best model and best AUC score
    """
    if not HAS_CATBOOST:
        return None, 0.0
    
    print(f"\n🔍 Optimizing CatBoost ({n_trials} trials)...")
    print("   This will take 30-45 minutes...")
    
    def objective(trial):
        params = {
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'random_seed': 42,
            'verbose': False,
            
            # Handle imbalanced data
            'auto_class_weights': 'Balanced',
            
            # Tree parameters
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10, log=True),
            
            # Regularization
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 5),
            'subsample': trial.suggest_float('subsample', 0.65, 0.95),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
        }
        
        # Create and train model
        model = CatBoostClassifier(
            **params,
            iterations=2000,
            early_stopping_rounds=50
        )
        
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )
        
        # Get validation score
        val_pred = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_pred)  # type: ignore[arg-type]
        
        return float(val_auc)
    
    # Run optimization
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)  # type: ignore[arg-type]
    
    # Train final model with best params
    best_params = study.best_params
    best_params.update({
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'auto_class_weights': 'Balanced',
        'random_seed': 42,
        'verbose': False
    })
    
    print(f"\n   Best CatBoost params found:")
    for key, value in sorted(study.best_params.items()):
        print(f"     {key}: {value}")
    
    # Train final model
    final_model = CatBoostClassifier(
        **best_params,
        iterations=2000,
        early_stopping_rounds=50
    )
    
    final_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=100
    )
    
    val_pred = final_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred)
    
    logger.info(f"✅ CatBoost optimized AUC: {val_auc:.4f}")
    
    return final_model, val_auc


def create_stacking_ensemble(models, X_val, y_val):
    """Create optimal ensemble weights for model predictions
    
    Args:
        models: Dictionary of models with their validation predictions
        X_val: Validation features
        y_val: Validation target
        
    Returns:
        Best ensemble weights and ensemble AUC
    """
    print("\n🎭 Creating stacking ensemble...")
    
    # Get predictions from each model
    predictions = {}
    
    if 'lightgbm' in models and models['lightgbm'] is not None:
        predictions['lightgbm'] = models['lightgbm'].predict(X_val)
    
    if 'xgboost' in models and models['xgboost'] is not None:
        dval = xgb.DMatrix(X_val)
        predictions['xgboost'] = models['xgboost'].predict(dval)
    
    if HAS_CATBOOST and 'catboost' in models and models['catboost'] is not None:
        predictions['catboost'] = models['catboost'].predict_proba(X_val)[:, 1]
    
    model_names = list(predictions.keys())
    n_models = len(model_names)
    
    if n_models == 1:
        # Only one model available
        logger.warning("Only one model available, no ensemble possible")
        return {model_names[0]: 1.0}, roc_auc_score(y_val, predictions[model_names[0]])
    
    elif n_models == 2:
        # Grid search for optimal 2-model weights
        print("   Finding optimal weights for 2-model ensemble...")
        best_auc = 0
        best_weight = 0.5
        
        for w in np.linspace(0.3, 0.7, 41):
            pred = w * predictions[model_names[0]] + (1 - w) * predictions[model_names[1]]
            auc = roc_auc_score(y_val, pred)
            
            if auc > best_auc:
                best_auc = auc
                best_weight = w
        
        weights = {model_names[0]: best_weight, model_names[1]: 1 - best_weight}
        
        print(f"   Best weights: {model_names[0]}={best_weight:.3f}, {model_names[1]}={1-best_weight:.3f}")
        
    else:
        # Use LogisticRegression meta-learner for 3+ models
        print("   Training LogisticRegression meta-learner for 3-model ensemble...")
        
        # Stack predictions
        stacked = np.column_stack([predictions[name] for name in model_names])
        
        # Train meta-learner
        meta = LogisticRegression(random_state=42, max_iter=1000)
        meta.fit(stacked, y_val)
        
        # Extract weights from coefficients
        raw_weights = np.abs(meta.coef_[0])
        normalized_weights = raw_weights / raw_weights.sum()
        
        weights = {name: float(w) for name, w in zip(model_names, normalized_weights)}
        
        print(f"   Learned weights:")
        for name, w in weights.items():
            print(f"     {name}: {w:.3f}")
        
        # Calculate ensemble prediction
        pred = sum(weights[name] * predictions[name] for name in model_names)
        best_auc = roc_auc_score(y_val, pred)
    
    # Calculate final ensemble
    ensemble_pred = sum(weights[name] * predictions[name] for name in model_names)
    ensemble_auc = roc_auc_score(y_val, ensemble_pred)
    
    logger.info(f"✅ Ensemble AUC: {ensemble_auc:.4f}")
    
    return weights, ensemble_auc


def main():
    """Main optimization pipeline"""
    
    print("=" * 80)
    print("🚀 Phase 3 COMPLETE: Optimization + Advanced Features")
    print("=" * 80)
    print(f"\n📊 Current AUC: 0.7885")
    print(f"🎯 Target AUC: 0.8000+")
    print(f"📈 Expected: 0.803-0.815 AUC")
    print(f"\n⏱️  Expected Duration: 2-3 hours")
    
    if HAS_CATBOOST:
        print(f"✅ Using 3-model ensemble (LightGBM + XGBoost + CatBoost)")
    else:
        print(f"⚠️  Using 2-model ensemble (LightGBM + XGBoost)")
    
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment("credscope_phase3_complete")
    
    # ===== STEP 1: Load Phase 2 Complete Features =====
    print(f"\n📁 Step 1/7: Loading Phase 2 Complete Features (522 features)...")
    print("-" * 80)
    
    feature_cache = Path(config['data']['features_path']) / 'engineered_features_COMPLETE.pkl'
    
    if not feature_cache.exists():
        logger.error(f"❌ Feature cache not found at {feature_cache}")
        logger.error("   Please run train_phase2_complete.py first!")
        return
    
    cached = joblib.load(feature_cache)
    train_df = cached['train']
    test_df = cached['test']
    feature_names = cached['features']
    
    logger.info(f"✅ Loaded {len(feature_names)} features from cache")
    logger.info(f"   Train shape: {train_df.shape}")
    logger.info(f"   Test shape: {test_df.shape}")
    
    # ===== STEP 2: Create Interaction Features =====
    print(f"\n🔧 Step 2/7: Creating Advanced Interaction Features (20 features)...")
    print("-" * 80)
    
    train_df = create_interaction_features(train_df, is_train=True)
    test_df = create_interaction_features(test_df, is_train=False)
    
    # Update feature names
    feature_cols = [col for col in train_df.columns if col != 'TARGET']
    logger.info(f"✅ Total features after interactions: {len(feature_cols)}")
    
    # ===== STEP 3: Prepare Train/Validation Split =====
    print(f"\n✂️ Step 3/7: Preparing Train/Validation Split...")
    print("-" * 80)
    
    X = train_df.drop(columns=['TARGET'])
    y = train_df['TARGET']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"   Train samples: {X_train.shape[0]:,}")
    logger.info(f"   Validation samples: {X_val.shape[0]:,}")
    logger.info(f"   Features: {X_train.shape[1]:,}")
    logger.info(f"   Class distribution - Train: {y_train.mean():.3%}, Val: {y_val.mean():.3%}")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"phase3_optimization_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        
        # Log parameters
        mlflow.log_param("base_features", len(feature_names))
        mlflow.log_param("total_features", X_train.shape[1])
        mlflow.log_param("has_catboost", HAS_CATBOOST)
        mlflow.log_param("n_trials_per_model", 150)
        
        models = {}
        scores = {}
        
        # ===== STEP 4: Optimize LightGBM =====
        print(f"\n🔍 Step 4/7: Optimizing LightGBM (150 trials)...")
        print("-" * 80)
        lgb_model, lgb_auc = optimize_lightgbm(X_train, y_train, X_val, y_val, n_trials=150)
        models['lightgbm'] = lgb_model
        scores['lightgbm'] = lgb_auc
        mlflow.log_metric("lightgbm_auc", float(lgb_auc))
        
        # ===== STEP 5: Optimize XGBoost =====
        print(f"\n🔍 Step 5/7: Optimizing XGBoost (150 trials)...")
        print("-" * 80)
        xgb_model, xgb_auc = optimize_xgboost(X_train, y_train, X_val, y_val, n_trials=150)
        models['xgboost'] = xgb_model
        scores['xgboost'] = xgb_auc
        mlflow.log_metric("xgboost_auc", float(xgb_auc))
        
        # ===== STEP 6: Optimize CatBoost (if available) =====
        if HAS_CATBOOST:
            print(f"\n🔍 Step 6/7: Optimizing CatBoost (150 trials)...")
            print("-" * 80)
            cb_model, cb_auc = optimize_catboost(X_train, y_train, X_val, y_val, n_trials=150)
            models['catboost'] = cb_model
            scores['catboost'] = cb_auc
            mlflow.log_metric("catboost_auc", float(cb_auc))
        else:
            print(f"\n⏭️  Step 6/7: Skipping CatBoost (not installed)...")
            print("-" * 80)
        
        # ===== STEP 7: Create Ensemble =====
        step_num = 7 if HAS_CATBOOST else 6
        print(f"\n🎭 Step {step_num}/7: Creating Stacking Ensemble...")
        print("-" * 80)
        
        ensemble_weights, ensemble_auc = create_stacking_ensemble(models, X_val, y_val)
        scores['ensemble'] = ensemble_auc
        mlflow.log_metric("ensemble_auc", float(ensemble_auc))
        
        # Log ensemble weights
        for model_name, weight in ensemble_weights.items():
            mlflow.log_param(f"ensemble_weight_{model_name}", float(weight))
        
        # Save models
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        # Save LightGBM
        if 'lightgbm' in models and models['lightgbm'] is not None:
            lgb_path = models_dir / 'lightgbm_phase3_optimized.txt'
            models['lightgbm'].save_model(str(lgb_path))
            logger.info(f"✅ Saved LightGBM model to {lgb_path}")
        
        # Save XGBoost
        if 'xgboost' in models and models['xgboost'] is not None:
            xgb_path = models_dir / 'xgboost_phase3_optimized.json'
            models['xgboost'].save_model(str(xgb_path))
            logger.info(f"✅ Saved XGBoost model to {xgb_path}")
        
        # Save CatBoost
        if HAS_CATBOOST and 'catboost' in models and models['catboost'] is not None:
            cb_path = models_dir / 'catboost_phase3_optimized.cbm'
            models['catboost'].save_model(str(cb_path))
            logger.info(f"✅ Saved CatBoost model to {cb_path}")
        
        # Save ensemble weights
        joblib.dump(ensemble_weights, models_dir / 'ensemble_weights_phase3.pkl')
        
        # Extract and save feature importance
        if 'lightgbm' in models and models['lightgbm'] is not None:
            importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': models['lightgbm'].feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False)
            
            importance_path = models_dir / 'feature_importance_phase3.csv'
            importance.to_csv(importance_path, index=False)
            logger.info(f"✅ Saved feature importance to {importance_path}")
            
            # Log top features
            print("\n📊 Top 10 Most Important Features:")
            print("-" * 80)
            for idx, (i, row) in enumerate(importance.head(10).iterrows()):  # type: ignore[operator]
                print(f"   {idx+1:2d}. {row['feature']:<40} {row['importance']:>10.0f}")
    
    # ===== FINAL SUMMARY =====
    print("\n" + "=" * 80)
    print("📊 FINAL RESULTS")
    print("=" * 80)
    
    print("\n🎯 Model Performance:")
    print("-" * 80)
    for model_name, auc in scores.items():
        if auc > 0:  # Only show models that were trained
            print(f"   {model_name.capitalize():<12} AUC: {auc:.4f}")
    
    print("\n📈 Performance Comparison:")
    print("-" * 80)
    comparisons = [
        ("Phase 1 Baseline", 0.7385, 0),
        ("Phase 2 Partial (4 tables)", 0.7794, 0.7385),
        ("Phase 2 Complete (7 tables)", 0.7885, 0.7794),
        ("Phase 3 Optimized", ensemble_auc, 0.7885)
    ]
    
    for name, current, previous in comparisons:
        if previous > 0:
            improvement = current - previous
            pct_improvement = (improvement / previous) * 100
            print(f"   {name:<30} AUC: {current:.4f}  (+{improvement:.4f}, +{pct_improvement:.2f}%)")
        else:
            print(f"   {name:<30} AUC: {current:.4f}")
    
    # Check if we hit the target
    print("\n" + "=" * 80)
    improvement_from_phase2 = ensemble_auc - 0.7885
    
    if ensemble_auc >= 0.82:
        print("🎉🎉🎉 EXCEPTIONAL! Exceeded 0.82 AUC!")
        print(f"   Improvement from Phase 2: +{improvement_from_phase2:.4f}")
    elif ensemble_auc >= 0.80:
        print("🎉🎉 SUCCESS! Achieved 0.80+ target!")
        print(f"   Improvement from Phase 2: +{improvement_from_phase2:.4f}")
    elif ensemble_auc > 0.7885:
        print(f"✅ Improved from Phase 2 by +{improvement_from_phase2:.4f}")
        remaining = 0.80 - ensemble_auc
        print(f"   Still {remaining:.4f} AUC short of 0.80 target")
    else:
        print("⚠️  No improvement from Phase 2")
        print("   Consider adjusting hyperparameter ranges or adding more features")
    
    print("\n💡 Next Steps:")
    if ensemble_auc >= 0.80:
        print("   ✅ Ready for Phase 4: Deployment!")
        print("   1. Create FastAPI service for model serving")
        print("   2. Build Streamlit dashboard for visualization")
        print("   3. Perform fairness analysis")
        print("   4. Generate final documentation")
    else:
        print("   🔧 Additional optimization needed:")
        print("   1. Try more Optuna trials (300-500)")
        print("   2. Add more interaction features")
        print("   3. Experiment with neural network models")
        print("   4. Try different ensemble strategies")
    
    print("\n" + "=" * 80)
    print("✅ Phase 3 Complete!")
    print("=" * 80)
    
    return scores


if __name__ == "__main__":
    results = main()