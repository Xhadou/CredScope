"""
Phase 3: Hyperparameter Optimization & Advanced Ensemble

This script uses Optuna for thorough hyperparameter optimization and builds
advanced ensemble models to achieve 0.82-0.83+ AUC.
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
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
import mlflow

# Our modules
from src.credscope.models.tuner import (
    LightGBMTuner,
    XGBoostTuner,
    optimize_ensemble_weights
)
from src.credscope.utils.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def load_cached_features(config: dict):
    """Load cached features from Phase 2
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_df, test_df, feature_names)
    """
    feature_cache = Path(config['data']['features_path']) / 'engineered_features.pkl'
    
    if not feature_cache.exists():
        raise FileNotFoundError(
            f"Cached features not found: {feature_cache}\n"
            f"Please run Phase 2 first: python scripts/train_phase2_features.py"
        )
    
    logger.info(f"Loading cached features from {feature_cache}")
    cached = joblib.load(feature_cache)
    
    logger.info(f"✓ Loaded {cached['train'].shape[0]:,} training samples")
    logger.info(f"✓ Loaded {cached['test'].shape[0]:,} test samples")
    logger.info(f"✓ Loaded {len(cached['features'])} features")
    
    return cached['train'], cached['test'], cached['features']


def create_stacking_ensemble(models_dict, X_train, y_train, X_val, y_val):
    """Create stacking ensemble with meta-learner
    
    Args:
        models_dict: Dictionary of {name: (model, params)}
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        Trained meta-learner and validation AUC
    """
    logger.info("🎭 Creating stacking ensemble with meta-learner...")
    
    # Get out-of-fold predictions for meta-learner
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Prepare arrays for OOF predictions
    oof_predictions = np.zeros((len(X_train), len(models_dict)))
    val_predictions = np.zeros((len(X_val), len(models_dict)))
    
    logger.info(f"Generating {n_folds}-fold out-of-fold predictions...")
    
    for idx, (model_name, (model, params)) in enumerate(models_dict.items()):
        logger.info(f"  Processing {model_name}...")
        
        oof_pred = np.zeros(len(X_train))
        val_pred_folds = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            
            # Train model on fold
            if 'lgb' in model_name.lower():
                # LightGBM
                train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
                fold_model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=params.get('n_estimators', 1000),
                    callbacks=[lgb.log_evaluation(period=0)]
                )
                oof_pred[val_idx] = fold_model.predict(X_fold_val)  # type: ignore[arg-type]
                val_pred_folds.append(fold_model.predict(X_val))  # type: ignore[arg-type]
                
            else:  # XGBoost
                dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
                dval_fold = xgb.DMatrix(X_fold_val)
                
                fold_model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=params.get('n_estimators', 1000),
                    verbose_eval=0
                )
                oof_pred[val_idx] = fold_model.predict(dval_fold)
                val_pred_folds.append(fold_model.predict(xgb.DMatrix(X_val)))
        
        oof_predictions[:, idx] = oof_pred
        val_predictions[:, idx] = np.mean(val_pred_folds, axis=0)
        
        oof_auc = roc_auc_score(y_train, oof_pred)
        logger.info(f"    OOF AUC: {oof_auc:.4f}")
    
    # Train meta-learner on OOF predictions
    logger.info("Training meta-learner (Logistic Regression)...")
    meta_model = LogisticRegression(random_state=42, max_iter=1000)
    meta_model.fit(oof_predictions, y_train)
    
    # Meta-learner predictions
    stacking_pred = meta_model.predict_proba(val_predictions)[:, 1]
    stacking_auc = roc_auc_score(y_val, stacking_pred)
    
    logger.info(f"✓ Stacking Ensemble AUC: {stacking_auc:.4f}")
    
    return meta_model, stacking_auc, val_predictions


def main():
    """Main Phase 3 training pipeline"""
    
    print("=" * 70)
    print("🚀 CredScope Phase 3: Hyperparameter Optimization & Advanced Ensemble")
    print("=" * 70)
    
    # Load config
    logger.info("Loading configuration...")
    config = load_config()
    
    # Initialize MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment("credscope_phase3")
    
    # Load cached features
    train_df, test_df, feature_names = load_cached_features(config)
    
    # Prepare data
    X = train_df.drop(columns=['TARGET'])
    y = train_df['TARGET']
    
    # Split data (same as Phase 2 for comparison)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    logger.info(f"\n📊 Data loaded:")
    logger.info(f"   Train: {X_train.shape[0]:,} samples")
    logger.info(f"   Validation: {X_val.shape[0]:,} samples")
    logger.info(f"   Features: {X_train.shape[1]:,}")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"phase3_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("optimization_trials", 200)
        
        # ===== STEP 1: OPTIMIZE LIGHTGBM =====
        print("\n" + "=" * 70)
        print("🔧 Step 1/4: Optimizing LightGBM (200 trials)")
        print("=" * 70)
        print("⏱️  Expected time: 15-25 minutes")
        print()
        
        lgb_tuner = LightGBMTuner(X_train, y_train, X_val, y_val, n_trials=200)
        lgb_best_params, lgb_best_score = lgb_tuner.optimize()
        lgb_optimized_model = lgb_tuner.get_best_model()
        
        # Log results
        mlflow.log_metric("lgb_optimized_auc", float(lgb_best_score))
        mlflow.log_params({f"lgb_{k}": v for k, v in lgb_best_params.items()})
        
        print(f"\n✅ LightGBM Optimization Complete!")
        print(f"   Phase 2 AUC: 0.7755")
        print(f"   Phase 3 AUC: {lgb_best_score:.4f}")
        print(f"   Improvement: +{lgb_best_score - 0.7755:.4f}")
        
        # ===== STEP 2: OPTIMIZE XGBOOST =====
        print("\n" + "=" * 70)
        print("🔧 Step 2/4: Optimizing XGBoost (200 trials)")
        print("=" * 70)
        print("⏱️  Expected time: 15-25 minutes")
        print()
        
        xgb_tuner = XGBoostTuner(X_train, y_train, X_val, y_val, n_trials=200)
        xgb_best_params, xgb_best_score = xgb_tuner.optimize()
        xgb_optimized_model = xgb_tuner.get_best_model()
        
        # Log results
        mlflow.log_metric("xgb_optimized_auc", float(xgb_best_score))
        mlflow.log_params({f"xgb_{k}": v for k, v in xgb_best_params.items()})
        
        print(f"\n✅ XGBoost Optimization Complete!")
        print(f"   Phase 2 AUC: 0.7785")
        print(f"   Phase 3 AUC: {xgb_best_score:.4f}")
        print(f"   Improvement: +{xgb_best_score - 0.7785:.4f}")
        
        # ===== STEP 3: WEIGHTED ENSEMBLE =====
        print("\n" + "=" * 70)
        print("🎭 Step 3/4: Optimizing Ensemble Weights")
        print("=" * 70)
        
        # Get predictions
        lgb_val_pred = lgb_optimized_model.predict(X_val)  # type: ignore[arg-type]
        xgb_val_pred = xgb_optimized_model.predict(xgb.DMatrix(X_val))
        
        # Optimize weights
        predictions_dict = {
            'lgb': lgb_val_pred,
            'xgb': xgb_val_pred
        }
        
        optimal_weights = optimize_ensemble_weights(predictions_dict, y_val, n_trials=100)
        
        # Calculate weighted ensemble
        weighted_ensemble_pred = (
            optimal_weights['lgb'] * lgb_val_pred +  # type: ignore[operator]
            optimal_weights['xgb'] * xgb_val_pred  # type: ignore[operator]
        )
        weighted_ensemble_auc = roc_auc_score(y_val, weighted_ensemble_pred)
        
        mlflow.log_metric("weighted_ensemble_auc", float(weighted_ensemble_auc))
        mlflow.log_params(optimal_weights)
        
        print(f"\n✅ Weighted Ensemble Complete!")
        print(f"   LightGBM weight: {optimal_weights['lgb']:.3f}")
        print(f"   XGBoost weight:  {optimal_weights['xgb']:.3f}")
        print(f"   Ensemble AUC: {weighted_ensemble_auc:.4f}")
        
        # ===== STEP 4: STACKING ENSEMBLE =====
        print("\n" + "=" * 70)
        print("🎭 Step 4/4: Creating Stacking Ensemble")
        print("=" * 70)
        print("⏱️  Expected time: 5-10 minutes")
        print()
        
        # Prepare params for stacking
        lgb_params_full = {
            **lgb_best_params,
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': 42
        }
        
        xgb_params_full = {
            **xgb_best_params,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'random_state': 42
        }
        
        models_dict = {
            'lgb': (lgb_optimized_model, lgb_params_full),
            'xgb': (xgb_optimized_model, xgb_params_full)
        }
        
        meta_model, stacking_auc, _ = create_stacking_ensemble(
            models_dict, X_train, y_train, X_val, y_val
        )
        
        mlflow.log_metric("stacking_ensemble_auc", float(stacking_auc))
        
        print(f"\n✅ Stacking Ensemble Complete!")
        print(f"   Stacking AUC: {stacking_auc:.4f}")
        
        # ===== FINAL COMPARISON =====
        print("\n" + "=" * 70)
        print("📊 PHASE 3 RESULTS SUMMARY")
        print("=" * 70)
        
        results = {
            'Baseline (Phase 1)': 0.7385,
            'Phase 2 - LightGBM': 0.7755,
            'Phase 2 - XGBoost': 0.7785,
            'Phase 2 - Simple Ensemble': 0.7794,
            'Phase 3 - Optimized LightGBM': lgb_best_score,
            'Phase 3 - Optimized XGBoost': xgb_best_score,
            'Phase 3 - Weighted Ensemble': weighted_ensemble_auc,
            'Phase 3 - Stacking Ensemble': stacking_auc
        }
        
        print("\n🎯 Performance Comparison:")
        print("-" * 70)
        for name, score in results.items():
            improvement = ((score - 0.7385) / 0.7385) * 100
            print(f"  {name:<35} AUC: {score:.4f}  (+{improvement:>5.2f}%)")
        
        # Best model
        best_model_name = max(results.items(), key=lambda x: x[1])[0]
        best_auc = max(results.values())
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   Final AUC: {best_auc:.4f}")
        print(f"   Total Improvement: +{best_auc - 0.7385:.4f} (+{((best_auc - 0.7385) / 0.7385) * 100:.2f}%)")
        
        # ===== SAVE MODELS =====
        print("\n" + "=" * 70)
        print("💾 Saving Optimized Models")
        print("=" * 70)
        
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True, parents=True)
        
        # Save LightGBM
        lgb_path = models_dir / 'lightgbm_phase3_optimized.txt'
        lgb_optimized_model.save_model(str(lgb_path))
        logger.info(f"✓ Optimized LightGBM saved to {lgb_path}")
        mlflow.log_artifact(str(lgb_path))
        
        # Save XGBoost
        xgb_path = models_dir / 'xgboost_phase3_optimized.json'
        xgb_optimized_model.save_model(str(xgb_path))
        logger.info(f"✓ Optimized XGBoost saved to {xgb_path}")
        mlflow.log_artifact(str(xgb_path))
        
        # Save best parameters
        params_path = models_dir / 'phase3_best_params.pkl'
        joblib.dump({
            'lgb_params': lgb_best_params,
            'xgb_params': xgb_best_params,
            'ensemble_weights': optimal_weights,
            'meta_model': meta_model
        }, params_path)
        logger.info(f"✓ Best parameters saved to {params_path}")
        mlflow.log_artifact(str(params_path))
        
        # Save results
        results_df = pd.DataFrame({
            'Model': list(results.keys()),
            'AUC': list(results.values())
        })
        results_path = models_dir / 'phase3_results.csv'
        results_df.to_csv(results_path, index=False)
        logger.info(f"✓ Results saved to {results_path}")
        mlflow.log_artifact(str(results_path))
        
        print("\n" + "=" * 70)
        print("✅ PHASE 3 COMPLETE!")
        print("=" * 70)
        
        print("\n💡 Next Steps:")
        print("  1. Run: mlflow ui --host 0.0.0.0")
        print("  2. View detailed metrics at http://localhost:5000")
        print("  3. Continue to Phase 4: SHAP Analysis & Model Interpretation")
        print("  4. Or proceed to Phase 5: Deployment (FastAPI + Streamlit)")
        
        print("\n" + "=" * 70)
        
        return {
            'lgb_auc': lgb_best_score,
            'xgb_auc': xgb_best_score,
            'weighted_ensemble_auc': weighted_ensemble_auc,
            'stacking_auc': stacking_auc,
            'best_auc': best_auc
        }


if __name__ == "__main__":
    results = main()