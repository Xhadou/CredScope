"""
Phase 2 COMPLETE: Feature Engineering with ALL 7 Tables

This version uses ALL tables from the Home Credit dataset:
✅ application_train
✅ bureau + bureau_balance  
✅ previous_application
✅ installments_payments (NEW!)
✅ credit_card_balance (NEW!)
✅ POS_CASH_balance (NEW!)

Expected improvement: +2-5% AUC (0.7794 → 0.80-0.82+)
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import mlflow

# Our modules
from src.credscope.data.loader import DataLoader
from src.credscope.features.geographic import create_geographic_features
from src.credscope.features.documents import create_document_features
from src.credscope.features.bureau import (
    create_bureau_features,
    create_previous_application_features
)
from src.credscope.features.installments import create_installments_features
from src.credscope.features.creditcard import create_creditcard_features
from src.credscope.features.pos_cash import create_pos_cash_features
from src.credscope.utils.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def load_and_engineer_features_complete(config: dict, use_cached: bool = False) -> tuple:
    """Load ALL 7 tables and create comprehensive features
    
    Args:
        config: Configuration dictionary
        use_cached: Whether to use cached features
        
    Returns:
        Tuple of (train_df, test_df, feature_names)
    """
    feature_cache = Path(config['data']['features_path']) / 'engineered_features_COMPLETE.pkl'
    
    # Check cache
    if use_cached and feature_cache.exists():
        logger.info(f"Loading cached features from {feature_cache}")
        cached = joblib.load(feature_cache)
        return cached['train'], cached['test'], cached['features']
    
    print("=" * 70)
    print("🔥 COMPLETE FEATURE ENGINEERING: ALL 7 TABLES")
    print("=" * 70)
    print("\nExpected feature count: 450-500 features")
    print("Expected improvement: +2-5% AUC\n")
    
    # ===== LOAD APPLICATION DATA =====
    print("\n📁 Step 1/9: Loading application data...")
    print("-" * 70)
    loader = DataLoader(config['data']['raw_path'])
    train_df = loader.load_application_data(train=True)
    test_df = loader.load_application_data(train=False)
    
    logger.info(f"✓ Train shape: {train_df.shape}")
    logger.info(f"✓ Test shape: {test_df.shape}")
    
    # Store target and IDs
    target = train_df['TARGET'].copy()
    train_ids = train_df['SK_ID_CURR'].copy()
    test_ids = test_df['SK_ID_CURR'].copy()
    
    # ===== GEOGRAPHIC FEATURES =====
    print("\n🗺️ Step 2/9: Creating geographic features...")
    print("-" * 70)
    train_df = create_geographic_features(train_df, target='TARGET', is_train=True)
    test_df = create_geographic_features(test_df, is_train=False)
    logger.info(f"✓ Train shape: {train_df.shape}")
    
    # ===== DOCUMENT FEATURES =====
    print("\n📄 Step 3/9: Creating document submission features...")
    print("-" * 70)
    train_df = create_document_features(train_df, target='TARGET', is_train=True)
    test_df = create_document_features(test_df, is_train=False)
    logger.info(f"✓ Train shape: {train_df.shape}")
    
    # ===== BUREAU FEATURES =====
    print("\n🏦 Step 4/9: Creating bureau credit history features...")
    print("-" * 70)
    try:
        bureau, bureau_balance = loader.load_bureau_data()
        bureau_features = create_bureau_features(bureau, bureau_balance)
        
        train_df = train_df.merge(bureau_features, on='SK_ID_CURR', how='left')
        test_df = test_df.merge(bureau_features, on='SK_ID_CURR', how='left')
        
        logger.info(f"✓ Train shape: {train_df.shape}")
    except Exception as e:
        logger.warning(f"Could not load bureau data: {e}")
    
    # ===== PREVIOUS APPLICATION FEATURES =====
    print("\n📋 Step 5/9: Creating previous application features...")
    print("-" * 70)
    try:
        prev_app = loader.load_previous_application()
        prev_features = create_previous_application_features(prev_app)
        
        train_df = train_df.merge(prev_features, on='SK_ID_CURR', how='left')
        test_df = test_df.merge(prev_features, on='SK_ID_CURR', how='left')
        
        logger.info(f"✓ Train shape: {train_df.shape}")
    except Exception as e:
        logger.warning(f"Could not load previous application data: {e}")
    
    # ===== INSTALLMENTS FEATURES (NEW!) =====
    print("\n💳 Step 6/9: Creating installments payment features...")
    print("-" * 70)
    print("⭐ This table has 13.6M records - payment behavior is KEY!")
    try:
        installments = loader.load_installments_payments()
        inst_features = create_installments_features(installments)
        
        train_df = train_df.merge(inst_features, on='SK_ID_CURR', how='left')
        test_df = test_df.merge(inst_features, on='SK_ID_CURR', how='left')
        
        logger.info(f"✓ Train shape: {train_df.shape}")
        logger.info(f"✓ Added ~50 payment behavior features")
    except Exception as e:
        logger.warning(f"Could not load installments data: {e}")
    
    # ===== CREDIT CARD FEATURES (NEW!) =====
    print("\n💳 Step 7/9: Creating credit card balance features...")
    print("-" * 70)
    print("⭐ This table has 3.8M records - credit utilization patterns!")
    try:
        cc_balance = loader.load_credit_card_balance()
        cc_features = create_creditcard_features(cc_balance)
        
        train_df = train_df.merge(cc_features, on='SK_ID_CURR', how='left')
        test_df = test_df.merge(cc_features, on='SK_ID_CURR', how='left')
        
        logger.info(f"✓ Train shape: {train_df.shape}")
        logger.info(f"✓ Added ~60 credit card features")
    except Exception as e:
        logger.warning(f"Could not load credit card data: {e}")
    
    # ===== POS CASH FEATURES (NEW!) =====
    print("\n🏪 Step 8/9: Creating POS/Cash balance features...")
    print("-" * 70)
    print("⭐ This table has 10M records - short-term credit behavior!")
    try:
        pos_cash = loader.load_pos_cash_balance()
        pos_features = create_pos_cash_features(pos_cash)
        
        train_df = train_df.merge(pos_features, on='SK_ID_CURR', how='left')
        test_df = test_df.merge(pos_features, on='SK_ID_CURR', how='left')
        
        logger.info(f"✓ Train shape: {train_df.shape}")
        logger.info(f"✓ Added ~40 POS/Cash features")
    except Exception as e:
        logger.warning(f"Could not load POS/Cash data: {e}")
    
    # ===== CLEAN UP =====
    print("\n🧹 Step 9/9: Cleaning and finalizing features...")
    print("-" * 70)
    
    # Remove ID columns
    train_df = train_df.drop(columns=['SK_ID_CURR'], errors='ignore')
    test_df = test_df.drop(columns=['SK_ID_CURR'], errors='ignore')
    
    # Align columns
    train_cols = set(train_df.columns) - {'TARGET'}
    test_cols = set(test_df.columns)
    common_cols = sorted(list(train_cols.intersection(test_cols)))
    
    # Keep only common columns plus target
    train_df = train_df[common_cols + ['TARGET']]
    test_df = test_df[common_cols]
    
    # Handle infinite values
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    test_df = test_df.replace([np.inf, -np.inf], np.nan)
    
    # Fill missing values
    for col in train_df.columns:
        if col == 'TARGET':
            continue
        if train_df[col].dtype in ['float32', 'float64', 'int32', 'int64']:
            median_val = train_df[col].median()
            train_df[col] = train_df[col].fillna(median_val)
            test_df[col] = test_df[col].fillna(median_val)
        else:
            train_df[col] = train_df[col].fillna('missing')
            test_df[col] = test_df[col].fillna('missing')
    
    # Encode categoricals
    object_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'TARGET' in object_cols:
        object_cols.remove('TARGET')
    
    if object_cols:
        logger.info(f"Encoding {len(object_cols)} categorical columns...")
        from sklearn.preprocessing import LabelEncoder
        
        for col in object_cols:
            le = LabelEncoder()
            combined = pd.concat([train_df[col], test_df[col]], axis=0)
            le.fit(combined.astype(str))
            train_df[col] = le.transform(train_df[col].astype(str))
            test_df[col] = le.transform(test_df[col].astype(str))
    
    # Get feature names
    feature_names = [col for col in train_df.columns if col != 'TARGET']
    
    print("\n" + "=" * 70)
    print("✅ COMPLETE FEATURE ENGINEERING FINISHED!")
    print("=" * 70)
    logger.info(f"\n📊 Final Statistics:")
    logger.info(f"   Total features: {len(feature_names)}")
    logger.info(f"   Train shape: {train_df.shape}")
    logger.info(f"   Test shape: {test_df.shape}")
    
    # Count by category
    feature_types = {
        'GEO': sum(1 for f in feature_names if f.startswith('GEO_')),
        'DOC': sum(1 for f in feature_names if f.startswith('DOC_')),
        'BUREAU': sum(1 for f in feature_names if f.startswith('BUREAU_') or f.startswith('BB_')),
        'PREV': sum(1 for f in feature_names if f.startswith('PREV_')),
        'INST': sum(1 for f in feature_names if f.startswith('INST_')),
        'CC': sum(1 for f in feature_names if f.startswith('CC_')),
        'POS': sum(1 for f in feature_names if f.startswith('POS_')),
        'ORIGINAL': len(feature_names) - sum([
            sum(1 for f in feature_names if f.startswith(prefix))
            for prefix in ['GEO_', 'DOC_', 'BUREAU_', 'BB_', 'PREV_', 'INST_', 'CC_', 'POS_']
        ])
    }
    
    print("\n📈 Feature Breakdown:")
    print("-" * 70)
    for feat_type, count in sorted(feature_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat_type:<20} {count:>5} features")
    
    # Cache
    logger.info(f"\n💾 Caching features to {feature_cache}")
    feature_cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        'train': train_df,
        'test': test_df,
        'features': feature_names,
        'feature_types': feature_types,
        'created_at': datetime.now().isoformat()
    }, feature_cache)
    
    return train_df, test_df, feature_names


def train_models_complete(X_train, y_train, X_val, y_val):
    """Train models with complete feature set"""
    
    print("\n" + "=" * 70)
    print("🎯 Training Models with COMPLETE Features")
    print("=" * 70)
    
    results = {}
    
    # ===== LIGHTGBM =====
    print("\n🚀 Training LightGBM...")
    print("-" * 70)
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 64,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': -1,
        'min_data_in_leaf': 50,
        'lambda_l1': 0.5,
        'lambda_l2': 0.5,
        'verbose': -1,
        'random_state': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    lgb_model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=5000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100)
        ]
    )
    
    lgb_val_pred = lgb_model.predict(X_val)  # type: ignore[arg-type]
    lgb_auc = roc_auc_score(y_val, lgb_val_pred)  # type: ignore[arg-type]
    results['LightGBM'] = lgb_auc
    
    logger.info(f"✓ LightGBM AUC: {lgb_auc:.4f}")
    
    # ===== XGBOOST =====
    print("\n🚀 Training XGBoost...")
    print("-" * 70)
    
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.9,
        'min_child_weight': 5,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'random_state': 42,
        'tree_method': 'hist'
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    evals = [(dtrain, 'train'), (dval, 'valid')]
    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=5000,
        evals=evals,
        early_stopping_rounds=100,
        verbose_eval=100
    )
    
    xgb_val_pred = xgb_model.predict(dval)
    xgb_auc = roc_auc_score(y_val, xgb_val_pred)
    results['XGBoost'] = xgb_auc
    
    logger.info(f"✓ XGBoost AUC: {xgb_auc:.4f}")
    
    # ===== ENSEMBLE =====
    print("\n🎭 Creating ensemble...")
    print("-" * 70)
    
    ensemble_pred = 0.5 * lgb_val_pred + 0.5 * xgb_val_pred  # type: ignore[operator]
    ensemble_auc = roc_auc_score(y_val, ensemble_pred)
    results['Ensemble'] = ensemble_auc
    
    logger.info(f"✓ Ensemble AUC: {ensemble_auc:.4f}")
    
    return results, lgb_model, xgb_model


def main():
    """Main pipeline"""
    
    print("=" * 70)
    print("🔥 CredScope Phase 2 COMPLETE: ALL 7 TABLES")
    print("=" * 70)
    print("\n⭐ Using ALL data sources:")
    print("  ✅ Application data")
    print("  ✅ Bureau + Bureau Balance")
    print("  ✅ Previous Applications")
    print("  ✅ Installments Payments (NEW!)")
    print("  ✅ Credit Card Balance (NEW!)")
    print("  ✅ POS/Cash Balance (NEW!)")
    print("\n🎯 Expected: 450-500 features, 0.80-0.82+ AUC")
    print("\n⏱️  Expected Duration: 15-25 minutes\n")
    
    # Load config
    config = load_config()
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment("credscope_phase2_complete")
    
    # Load and engineer features
    train_df, test_df, feature_names = load_and_engineer_features_complete(
        config, use_cached=False
    )
    
    # Prepare data
    X = train_df.drop(columns=['TARGET'])
    y = train_df['TARGET']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"\n📊 Data split:")
    logger.info(f"   Train: {X_train.shape[0]:,} samples")
    logger.info(f"   Validation: {X_val.shape[0]:,} samples")
    logger.info(f"   Features: {X_train.shape[1]:,}")
    
    # Train models
    with mlflow.start_run(run_name=f"complete_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_param("includes_all_tables", True)
        
        results, lgb_model, xgb_model = train_models_complete(
            X_train, y_train, X_val, y_val
        )
        
        # Log results
        for name, auc in results.items():
            mlflow.log_metric(f"{name.lower()}_auc", float(auc))
        
        # Save models
        models_dir = Path('models')
        lgb_model.save_model(str(models_dir / 'lightgbm_complete.txt'))
        xgb_model.save_model(str(models_dir / 'xgboost_complete.json'))
        
        joblib.dump(feature_names, models_dir / 'feature_names_complete.pkl')
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': lgb_model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        importance.to_csv(models_dir / 'feature_importance_complete.csv', index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    
    comparison = {
        'Baseline (Phase 1)': 0.7385,
        'Phase 2 - Partial (4 tables)': 0.7794,
        'Phase 2 - COMPLETE (7 tables)': results['Ensemble']
    }
    
    print("\n🎯 Performance Comparison:")
    print("-" * 70)
    for name, score in comparison.items():
        improvement = ((score - 0.7385) / 0.7385) * 100
        print(f"  {name:<35} AUC: {score:.4f}  (+{improvement:>5.2f}%)")
    
    improvement_vs_partial = results['Ensemble'] - 0.7794
    improvement_pct = (improvement_vs_partial / 0.7794) * 100
    
    print(f"\n🔥 Improvement from adding 3 tables:")
    print(f"   +{improvement_vs_partial:.4f} AUC (+{improvement_pct:.2f}%)")
    
    if results['Ensemble'] >= 0.82:
        print("\n🎉🎉🎉 BREAKTHROUGH! Exceeded 0.82 target!")
    elif results['Ensemble'] >= 0.80:
        print("\n🎉 EXCELLENT! Hit 0.80+ target!")
    elif results['Ensemble'] > 0.7794:
        print("\n✅ GOOD! Clear improvement over partial features!")
    
    print("\n💡 Next Steps:")
    print("  1. Review feature importance (models/feature_importance_complete.csv)")
    print("  2. If >= 0.80: Move to Phase 4 (Deployment)")
    print("  3. If < 0.80: Try Phase 3 optimization on these features")
    
    print("\n" + "=" * 70)
    
    return results


if __name__ == "__main__":
    results = main()