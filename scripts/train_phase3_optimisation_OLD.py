"""
Phase 3 Enhanced: Aggressive Optimization with Feature Selection

This script uses more aggressive strategies to push beyond Phase 2 performance:
- Wider hyperparameter search spaces
- Feature importance-based selection
- Multiple ensemble strategies
- Calibration techniques
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
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import mlflow

# Our modules
from src.credscope.utils.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def load_cached_features(config: dict) -> tuple:
    """Load cached features from Phase 2"""
    feature_cache = Path(config['data']['features_path']) / 'engineered_features.pkl'
    
    if not feature_cache.exists():
        raise FileNotFoundError(f"Cached features not found: {feature_cache}")
    
    logger.info(f"Loading cached features from {feature_cache}")
    cached = joblib.load(feature_cache)
    
    return cached['train'], cached['test'], cached['features']


def select_top_features(X_train, y_train, X_val, n_features: int = 200):
    """Select top N most important features using LightGBM
    
    Args:
        X_train, y_train: Training data
        X_val: Validation data
        n_features: Number of features to keep
        
    Returns:
        Selected feature names, X_train_selected, X_val_selected
    """
    logger.info(f"Selecting top {n_features} features using feature importance...")
    
    # Train quick LightGBM to get feature importance
    quick_model = lgb.LGBMClassifier(
        n_estimators=100,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    )
    quick_model.fit(X_train, y_train)
    
    # Get feature importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': quick_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Select top features
    top_features = importance.head(n_features)['feature'].tolist()
    
    logger.info(f"✓ Selected {len(top_features)} features")
    logger.info(f"  Top 5: {top_features[:5]}")
    
    return top_features, X_train[top_features], X_val[top_features]


def optimize_lightgbm_aggressive(X_train, y_train, n_trials: int = 150):
    """Aggressive LightGBM optimization with wider search space"""
    
    logger.info("🔥 Starting AGGRESSIVE LightGBM optimization...")
    
    def objective(trial):
        # WIDER search space than before
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'random_state': 42,
            'force_col_wise': True,
            
            # More aggressive tree structure
            'num_leaves': trial.suggest_int('num_leaves', 10, 200),  # Wider range
            'max_depth': trial.suggest_int('max_depth', 2, 15),      # Deeper trees
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),  # More flexible
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-4, 100.0, log=True),
            
            # Learning rate variations
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),  # Wider
            'n_estimators': trial.suggest_int('n_estimators', 200, 3000),  # More trees
            
            # Aggressive sampling
            'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 1.0),  # More variation
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.3, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            
            # Stronger regularization options
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 100.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 100.0, log=True),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0, 20),
            
            # Additional parameters
            'path_smooth': trial.suggest_float('path_smooth', 0, 1),
            'max_bin': trial.suggest_int('max_bin', 100, 500),
        }
        
        # 3-fold CV (faster than 5-fold)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr = X_train.iloc[train_idx] if isinstance(X_train, pd.DataFrame) else X_train[train_idx]
            X_va = X_train.iloc[val_idx] if isinstance(X_train, pd.DataFrame) else X_train[val_idx]
            y_tr = y_train.iloc[train_idx] if isinstance(y_train, pd.Series) else y_train[train_idx]
            y_va = y_train.iloc[val_idx] if isinstance(y_train, pd.Series) else y_train[val_idx]
            
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(0)
                ]
            )
            
            y_pred = model.predict_proba(X_va)[:, 1]  # type: ignore
            score = roc_auc_score(y_va, y_pred)
            scores.append(score)
        
        return float(np.mean(scores))
    
    # Create study with more aggressive sampling
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42, n_startup_trials=20)  # More exploration
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"✓ Best LightGBM AUC: {study.best_value:.4f}")
    logger.info(f"  Best params: {study.best_params}")
    
    return study.best_params, study.best_value


def optimize_xgboost_aggressive(X_train, y_train, n_trials: int = 150):
    """Aggressive XGBoost optimization with wider search space"""
    
    logger.info("🔥 Starting AGGRESSIVE XGBoost optimization...")
    
    def objective(trial):
        # WIDER search space
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'random_state': 42,
            'verbosity': 0,
            
            # More aggressive tree structure
            'max_depth': trial.suggest_int('max_depth', 2, 12),  # Deeper
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 500),  # Wider
            'gamma': trial.suggest_float('gamma', 1e-8, 10.0, log=True),
            
            # Learning variations
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 200, 3000),
            
            # Aggressive sampling
            'subsample': trial.suggest_float('subsample', 0.3, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.3, 1.0),
            
            # Stronger regularization
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100.0, log=True),
            
            # Additional parameters
            'max_bin': trial.suggest_int('max_bin', 100, 500),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 20),  # Handle imbalance
        }
        
        # 3-fold CV
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr = X_train.iloc[train_idx] if isinstance(X_train, pd.DataFrame) else X_train[train_idx]
            X_va = X_train.iloc[val_idx] if isinstance(X_train, pd.DataFrame) else X_train[val_idx]
            y_tr = y_train.iloc[train_idx] if isinstance(y_train, pd.Series) else y_train[train_idx]
            y_va = y_train.iloc[val_idx] if isinstance(y_train, pd.Series) else y_train[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            y_pred = model.predict_proba(X_va)[:, 1]  # type: ignore
            score = roc_auc_score(y_va, y_pred)
            scores.append(score)
        
        return float(np.mean(scores))
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42, n_startup_trials=20)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"✓ Best XGBoost AUC: {study.best_value:.4f}")
    logger.info(f"  Best params: {study.best_params}")
    
    return study.best_params, study.best_value


def create_calibrated_models(lgb_model, xgb_model, X_train, y_train):
    """Create probability-calibrated versions of models"""
    
    logger.info("🎯 Creating calibrated models...")
    
    # Calibrate LightGBM
    lgb_calibrated = CalibratedClassifierCV(
        lgb_model,
        method='isotonic',
        cv=3
    )
    lgb_calibrated.fit(X_train, y_train)
    
    # Calibrate XGBoost  
    xgb_calibrated = CalibratedClassifierCV(
        xgb_model,
        method='isotonic',
        cv=3
    )
    xgb_calibrated.fit(X_train, y_train)
    
    logger.info("✓ Calibration complete")
    
    return lgb_calibrated, xgb_calibrated


def create_voting_ensemble(lgb_model, xgb_model, X_train, y_train, X_val, y_val):
    """Create sklearn VotingClassifier ensemble"""
    
    logger.info("🗳️ Creating voting ensemble...")
    
    # Try different weight combinations
    best_auc = 0
    best_weights = None
    
    for w1 in [0.3, 0.4, 0.5, 0.6, 0.7]:
        w2 = 1 - w1
        
        voting = VotingClassifier(
            estimators=[('lgb', lgb_model), ('xgb', xgb_model)],
            voting='soft',
            weights=[w1, w2]
        )
        voting.fit(X_train, y_train)
        
        y_pred = voting.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        
        if auc > best_auc:
            best_auc = auc
            best_weights = [w1, w2]
    
    logger.info(f"✓ Best voting ensemble: {best_weights} → AUC: {best_auc:.4f}")
    
    # Create final voting ensemble
    final_voting = VotingClassifier(
        estimators=[('lgb', lgb_model), ('xgb', xgb_model)],
        voting='soft',
        weights=best_weights
    )
    final_voting.fit(X_train, y_train)
    
    return final_voting, best_auc


def create_blending_ensemble(models_dict, X_val, y_val):
    """Create blended predictions with optimized weights"""
    
    logger.info("🎨 Creating blended ensemble...")
    
    # Get predictions from all models
    predictions = {}
    for name, model in models_dict.items():
        if hasattr(model, 'predict_proba'):
            pred = model.predict_proba(X_val)[:, 1]
        else:
            pred = model.predict(X_val)
        predictions[name] = pred
    
    # Optimize weights with Optuna
    def objective(trial):
        weights = []
        remaining = 1.0
        
        for i in range(len(models_dict) - 1):
            w = trial.suggest_float(f'w{i}', 0.0, remaining)
            weights.append(w)
            remaining -= w
        weights.append(remaining)
        
        # Weighted average
        blend = sum(w * predictions[name] for w, name in zip(weights, predictions.keys()))
        return float(roc_auc_score(y_val, blend))
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, show_progress_bar=False)
    
    # Get best weights
    best_weights = []
    remaining = 1.0
    for i in range(len(models_dict) - 1):
        w = study.best_params[f'w{i}']
        best_weights.append(w)
        remaining -= w
    best_weights.append(remaining)
    
    # Create final blend
    final_blend = sum(w * predictions[name] for w, name in zip(best_weights, predictions.keys()))
    blend_auc = roc_auc_score(y_val, final_blend)
    
    logger.info(f"✓ Blended ensemble AUC: {blend_auc:.4f}")
    logger.info(f"  Weights: {dict(zip(predictions.keys(), best_weights))}")
    
    return final_blend, blend_auc, dict(zip(predictions.keys(), best_weights))


def main():
    """Main enhanced optimization pipeline"""
    
    print("=" * 70)
    print("🔥 CredScope Phase 3 ENHANCED: Aggressive Optimization")
    print("=" * 70)
    print("\n💡 Strategy:")
    print("  ✓ Feature selection (top 200 features)")
    print("  ✓ Wider hyperparameter search spaces")
    print("  ✓ Probability calibration")
    print("  ✓ Multiple ensemble strategies")
    print("  ✓ 3-fold CV (faster, more trials)")
    print("\n⏱️  Expected Duration: 30-45 minutes\n")
    
    # Load config
    config = load_config()
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment("credscope_phase3_enhanced")
    
    # Load features
    print("=" * 70)
    print("📁 Loading Phase 2 features")
    print("=" * 70)
    
    train_df, test_df, feature_names = load_cached_features(config)
    
    X = train_df.drop(columns=['TARGET'])
    y = train_df['TARGET']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"✓ Loaded data: {X_train.shape[0]:,} train, {X_val.shape[0]:,} val")
    
    with mlflow.start_run(run_name=f"enhanced_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        
        # ===== FEATURE SELECTION =====
        print("\n" + "=" * 70)
        print("🎯 Step 1/5: Feature Selection")
        print("=" * 70)
        
        top_features, X_train_selected, X_val_selected = select_top_features(
            X_train, y_train, X_val, n_features=200
        )
        
        mlflow.log_param("n_features_selected", len(top_features))
        
        # ===== OPTIMIZE LIGHTGBM =====
        print("\n" + "=" * 70)
        print("🔥 Step 2/5: Aggressive LightGBM Optimization (150 trials)")
        print("=" * 70)
        
        lgb_params, lgb_cv_score = optimize_lightgbm_aggressive(
            X_train_selected, y_train, n_trials=150
        )
        
        # Train final LightGBM
        lgb_final = lgb.LGBMClassifier(**lgb_params)
        lgb_final.fit(X_train_selected, y_train)
        lgb_val_pred = lgb_final.predict_proba(X_val_selected)[:, 1]  # type: ignore
        lgb_val_auc = roc_auc_score(y_val, lgb_val_pred)
        
        logger.info(f"✓ LightGBM Val AUC: {lgb_val_auc:.4f}")
        mlflow.log_metric("lgb_val_auc", float(lgb_val_auc))
        
        # ===== OPTIMIZE XGBOOST =====
        print("\n" + "=" * 70)
        print("🔥 Step 3/5: Aggressive XGBoost Optimization (150 trials)")
        print("=" * 70)
        
        xgb_params, xgb_cv_score = optimize_xgboost_aggressive(
            X_train_selected, y_train, n_trials=150
        )
        
        # Train final XGBoost
        xgb_final = xgb.XGBClassifier(**xgb_params)
        xgb_final.fit(X_train_selected, y_train)
        xgb_val_pred = xgb_final.predict_proba(X_val_selected)[:, 1]  # type: ignore
        xgb_val_auc = roc_auc_score(y_val, xgb_val_pred)
        
        logger.info(f"✓ XGBoost Val AUC: {xgb_val_auc:.4f}")
        mlflow.log_metric("xgb_val_auc", float(xgb_val_auc))
        
        # ===== CALIBRATION =====
        print("\n" + "=" * 70)
        print("🎯 Step 4/5: Probability Calibration")
        print("=" * 70)
        
        lgb_calibrated, xgb_calibrated = create_calibrated_models(
            lgb_final, xgb_final, X_train_selected, y_train
        )
        
        lgb_cal_pred = lgb_calibrated.predict_proba(X_val_selected)[:, 1]
        xgb_cal_pred = xgb_calibrated.predict_proba(X_val_selected)[:, 1]
        
        lgb_cal_auc = roc_auc_score(y_val, lgb_cal_pred)
        xgb_cal_auc = roc_auc_score(y_val, xgb_cal_pred)
        
        logger.info(f"✓ Calibrated LightGBM AUC: {lgb_cal_auc:.4f}")
        logger.info(f"✓ Calibrated XGBoost AUC: {xgb_cal_auc:.4f}")
        mlflow.log_metric("lgb_calibrated_auc", float(lgb_cal_auc))
        mlflow.log_metric("xgb_calibrated_auc", float(xgb_cal_auc))
        
        # ===== ENSEMBLE STRATEGIES =====
        print("\n" + "=" * 70)
        print("🎭 Step 5/5: Multiple Ensemble Strategies")
        print("=" * 70)
        
        # Strategy 1: Voting Ensemble
        voting, voting_auc = create_voting_ensemble(
            lgb_final, xgb_final, X_train_selected, y_train, X_val_selected, y_val
        )
        mlflow.log_metric("voting_auc", float(voting_auc))
        
        # Strategy 2: Blended Ensemble (all models)
        all_models = {
            'lgb': lgb_final,
            'xgb': xgb_final,
            'lgb_cal': lgb_calibrated,
            'xgb_cal': xgb_calibrated
        }
        
        blend_pred, blend_auc, blend_weights = create_blending_ensemble(
            all_models, X_val_selected, y_val
        )
        mlflow.log_metric("blend_auc", float(blend_auc))
        
        # Strategy 3: Simple Average
        simple_avg = (lgb_val_pred + xgb_val_pred) / 2
        simple_avg_auc = roc_auc_score(y_val, simple_avg)
        mlflow.log_metric("simple_avg_auc", float(simple_avg_auc))
        
        # ===== RESULTS =====
        print("\n" + "=" * 70)
        print("📊 ENHANCED PHASE 3 RESULTS")
        print("=" * 70)
        
        results = {
            'Baseline (Phase 1)': 0.7385,
            'Phase 2 - LightGBM': 0.7755,
            'Phase 2 - XGBoost': 0.7785,
            'Phase 2 - Simple Ensemble': 0.7794,
            'Phase 3 - Optimized LightGBM': lgb_val_auc,
            'Phase 3 - Optimized XGBoost': xgb_val_auc,
            'Phase 3 - Calibrated LightGBM': lgb_cal_auc,
            'Phase 3 - Calibrated XGBoost': xgb_cal_auc,
            'Phase 3 - Voting Ensemble': voting_auc,
            'Phase 3 - Simple Average': simple_avg_auc,
            'Phase 3 - Blended Ensemble': blend_auc,
        }
        
        print("\n🎯 Performance Comparison:")
        print("-" * 70)
        for name, score in results.items():
            improvement = ((score - 0.7385) / 0.7385) * 100
            vs_phase2 = ((score - 0.7794) / 0.7794) * 100 if score > 0.7794 else 0
            print(f"  {name:<40} AUC: {score:.4f}  (+{improvement:>5.2f}%)")
            if vs_phase2 > 0:
                print(f"  {'':<40}          vs Phase2: +{vs_phase2:.2f}%")
        
        # Best model
        best_name = max(results.items(), key=lambda x: x[1])[0]
        best_auc = max(results.values())
        
        print(f"\n🏆 Best Model: {best_name}")
        print(f"   Final AUC: {best_auc:.4f}")
        print(f"   Improvement vs Baseline: +{best_auc - 0.7385:.4f} (+{((best_auc - 0.7385) / 0.7385) * 100:.2f}%)")
        print(f"   Improvement vs Phase 2: +{best_auc - 0.7794:.4f} (+{((best_auc - 0.7794) / 0.7794) * 100:.2f}%)")
        
        # ===== SAVE MODELS =====
        print("\n" + "=" * 70)
        print("💾 Saving Models")
        print("=" * 70)
        
        models_dir = Path('models')
        
        # Save best individual models
        joblib.dump(lgb_calibrated, models_dir / 'lgb_calibrated_enhanced.pkl')
        joblib.dump(xgb_calibrated, models_dir / 'xgb_calibrated_enhanced.pkl')
        joblib.dump(voting, models_dir / 'voting_ensemble_enhanced.pkl')
        
        # Save metadata
        metadata = {
            'best_lgb_params': lgb_params,
            'best_xgb_params': xgb_params,
            'blend_weights': blend_weights,
            'selected_features': top_features,
            'results': results,
            'best_model': best_name,
            'best_auc': best_auc
        }
        joblib.dump(metadata, models_dir / 'phase3_enhanced_metadata.pkl')
        
        # Save results CSV
        results_df = pd.DataFrame({
            'Model': list(results.keys()),
            'AUC': list(results.values())
        })
        results_df.to_csv(models_dir / 'phase3_enhanced_results.csv', index=False)
        
        logger.info("✓ All models and metadata saved")
        
        print("\n" + "=" * 70)
        print("✅ ENHANCED PHASE 3 COMPLETE!")
        print("=" * 70)
        
        if best_auc >= 0.82:
            print("\n🎊 🏆 EXCELLENT! Target exceeded! (≥0.82 AUC)")
        elif best_auc >= 0.80:
            print("\n🎯 GREAT! Strong improvement achieved!")
        elif best_auc > 0.7794:
            print("\n📈 GOOD! We improved over Phase 2!")
        else:
            print("\n💡 Model may be at ceiling with current features")
            print("   Consider: Feature engineering, external data, different algorithms")
        
        print(f"\n💡 Next Steps:")
        print(f"  1. Review feature importance of selected features")
        print(f"  2. Analyze predictions for insights")
        print(f"  3. Consider additional feature engineering if needed")
        print(f"  4. Move to deployment if satisfied with performance")
        
        return {
            'best_auc': best_auc,
            'improvement_vs_phase2': best_auc - 0.7794,
            'all_results': results
        }


if __name__ == "__main__":
    results = main()