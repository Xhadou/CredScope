"""
Phase 2: Feature Engineering and Enhanced Model Training

This script orchestrates the complete feature engineering pipeline and trains
enhanced models with alternative credit features.
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
from sklearn.metrics import roc_auc_score, classification_report
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
from src.credscope.utils.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def load_and_engineer_features(config: dict, 
                               use_cached: bool = True) -> tuple:
    """Load data and create all engineered features
    
    Args:
        config: Configuration dictionary
        use_cached: Whether to use cached features if available
        
    Returns:
        Tuple of (train_df, test_df, feature_names)
    """
    feature_cache = Path(config['data']['features_path']) / 'engineered_features.pkl'
    
    # Check cache
    if use_cached and feature_cache.exists():
        logger.info(f"Loading cached features from {feature_cache}")
        cached = joblib.load(feature_cache)
        return cached['train'], cached['test'], cached['features']
    
    logger.info("🔧 Starting feature engineering pipeline...")
    
    # ===== LOAD APPLICATION DATA =====
    logger.info("\n📁 Step 1/6: Loading application data...")
    loader = DataLoader(config['data']['raw_path'])
    train_df = loader.load_application_data(train=True)
    test_df = loader.load_application_data(train=False)
    
    logger.info(f"✓ Train shape: {train_df.shape}")
    logger.info(f"✓ Test shape: {test_df.shape}")
    
    # Store target
    target = train_df['TARGET'].copy()
    train_ids = train_df['SK_ID_CURR'].copy()
    test_ids = test_df['SK_ID_CURR'].copy()
    
    # ===== GEOGRAPHIC FEATURES =====
    logger.info("\n🗺️ Step 2/6: Creating geographic features...")
    train_df = create_geographic_features(train_df, target='TARGET', is_train=True)
    test_df = create_geographic_features(test_df, is_train=False)
    logger.info(f"✓ Train shape after geographic features: {train_df.shape}")
    
    # ===== DOCUMENT FEATURES =====
    logger.info("\n📄 Step 3/6: Creating document submission features...")
    train_df = create_document_features(train_df, target='TARGET', is_train=True)
    test_df = create_document_features(test_df, is_train=False)
    logger.info(f"✓ Train shape after document features: {train_df.shape}")
    
    # ===== BUREAU FEATURES =====
    logger.info("\n🏦 Step 4/6: Creating bureau credit history features...")
    try:
        bureau, bureau_balance = loader.load_bureau_data()
        
        bureau_features = create_bureau_features(bureau, bureau_balance)
        
        # Merge with main dataframes
        train_df = train_df.merge(bureau_features, on='SK_ID_CURR', how='left')
        test_df = test_df.merge(bureau_features, on='SK_ID_CURR', how='left')
        
        logger.info(f"✓ Train shape after bureau features: {train_df.shape}")
    except Exception as e:
        logger.warning(f"Could not load bureau data: {e}")
    
    # ===== PREVIOUS APPLICATION FEATURES =====
    logger.info("\n📋 Step 5/6: Creating previous application features...")
    try:
        prev_app = loader.load_previous_application()
        prev_features = create_previous_application_features(prev_app)
        
        # Merge with main dataframes
        train_df = train_df.merge(prev_features, on='SK_ID_CURR', how='left')
        test_df = test_df.merge(prev_features, on='SK_ID_CURR', how='left')
        
        logger.info(f"✓ Train shape after previous application features: {train_df.shape}")
    except Exception as e:
        logger.warning(f"Could not load previous application data: {e}")
    
    # ===== CLEAN UP =====
    logger.info("\n🧹 Step 6/6: Cleaning and finalizing features...")
    
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
    
    # Fill missing values (simple strategy for now)
    for col in train_df.columns:
        if col == 'TARGET':
            continue
        
        col_dtype = str(train_df[col].dtype)
        
        # Handle numeric columns
        if train_df[col].dtype in ['float32', 'float64', 'int32', 'int64', 'float', 'int']:
            # Fill with median
            median_val = train_df[col].median()
            train_df[col] = train_df[col].fillna(median_val)
            test_df[col] = test_df[col].fillna(median_val)
        
        # Handle categorical columns
        elif 'category' in col_dtype:
            # Add 'missing' to categories first
            if 'missing' not in train_df[col].cat.categories:
                train_df[col] = train_df[col].cat.add_categories(['missing'])
            if 'missing' not in test_df[col].cat.categories:
                test_df[col] = test_df[col].cat.add_categories(['missing'])
            
            # Now fill with 'missing'
            train_df[col] = train_df[col].fillna('missing')
            test_df[col] = test_df[col].fillna('missing')
        
        # Handle object/string columns
        else:
            # Fill with mode or 'missing'
            if train_df[col].notna().sum() > 0:
                mode_val = train_df[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else 'missing'
            else:
                fill_val = 'missing'
            
            train_df[col] = train_df[col].fillna(fill_val)
            test_df[col] = test_df[col].fillna(fill_val)
    
    # ===== ENCODE CATEGORICAL VARIABLES =====
    logger.info("\n🔢 Encoding categorical variables...")
    
    # Identify object/categorical columns
    categorical_cols = []
    for col in train_df.columns:
        if col == 'TARGET':
            continue
        if train_df[col].dtype == 'object' or str(train_df[col].dtype) == 'category':
            categorical_cols.append(col)
    
    logger.info(f"Found {len(categorical_cols)} categorical columns to encode")
    
    # Label encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    
    for col in categorical_cols:
        le = LabelEncoder()
        
        # Combine train and test to ensure consistent encoding
        combined = pd.concat([train_df[col], test_df[col]], axis=0)
        le.fit(combined.astype(str))
        
        # Transform both sets
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
    
    logger.info(f"✓ Encoded {len(categorical_cols)} categorical columns")
    
    # Get feature names
    feature_names = [col for col in train_df.columns if col != 'TARGET']
    
    logger.info(f"\n✅ Feature engineering complete!")
    logger.info(f"   Total features: {len(feature_names)}")
    logger.info(f"   Train shape: {train_df.shape}")
    logger.info(f"   Test shape: {test_df.shape}")
    
    # Cache the features
    logger.info(f"\n💾 Caching features to {feature_cache}")
    feature_cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        'train': train_df,
        'test': test_df,
        'features': feature_names,
        'created_at': datetime.now().isoformat()
    }, feature_cache)
    
    return train_df, test_df, feature_names


def train_lightgbm_model(X_train, y_train, X_val, y_val, params=None):
    """Train LightGBM model with custom parameters
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        params: Model parameters (optional)
        
    Returns:
        Trained model
    """
    logger.info("🚀 Training LightGBM model...")
    
    if params is None:
        params = {
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
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train
    model = lgb.train(
        params,
        train_data,
        num_boost_round=5000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # Predictions
    train_pred = model.predict(X_train)  # type: ignore[arg-type]
    val_pred = model.predict(X_val)  # type: ignore[arg-type]
    
    train_auc = roc_auc_score(y_train, train_pred)  # type: ignore[arg-type]
    val_auc = roc_auc_score(y_val, val_pred)  # type: ignore[arg-type]
    
    logger.info(f"✓ LightGBM - Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
    
    return model, val_auc


def train_xgboost_model(X_train, y_train, X_val, y_val, params=None):
    """Train XGBoost model
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        params: Model parameters (optional)
        
    Returns:
        Trained model
    """
    logger.info("🚀 Training XGBoost model...")
    
    if params is None:
        params = {
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
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Train
    evals = [(dtrain, 'train'), (dval, 'valid')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        evals=evals,
        early_stopping_rounds=100,
        verbose_eval=100
    )
    
    # Predictions
    train_pred = model.predict(dtrain)
    val_pred = model.predict(dval)
    
    train_auc = roc_auc_score(y_train, train_pred)
    val_auc = roc_auc_score(y_val, val_pred)
    
    logger.info(f"✓ XGBoost - Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
    
    return model, val_auc


def main():
    """Main training pipeline"""
    
    print("=" * 60)
    print("🚀 CredScope Phase 2: Feature Engineering & Enhanced Models")
    print("=" * 60)
    
    # Load config
    logger.info("Loading configuration...")
    config = load_config()
    
    # Initialize MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment("credscope_phase2")
    
    # Load and engineer features
    train_df, test_df, feature_names = load_and_engineer_features(config, use_cached=False)
    
    # Prepare data for training
    X = train_df.drop(columns=['TARGET'])
    y = train_df['TARGET']
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    logger.info(f"\n📊 Data split:")
    logger.info(f"   Train: {X_train.shape[0]:,} samples")
    logger.info(f"   Validation: {X_val.shape[0]:,} samples")
    logger.info(f"   Features: {X_train.shape[1]:,}")
    logger.info(f"   Default rate - Train: {y_train.mean():.2%}, Val: {y_val.mean():.2%}")
    
    # ===== TRAIN MODELS =====
    print("\n" + "=" * 60)
    print("🎯 Training Enhanced Models")
    print("=" * 60)
    
    with mlflow.start_run(run_name=f"phase2_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log parameters
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("val_samples", len(X_val))
        
        # Train LightGBM
        print("\n" + "-" * 60)
        lgb_model, lgb_auc = train_lightgbm_model(X_train, y_train, X_val, y_val)
        
        # Log LightGBM metrics
        mlflow.log_metric("lgb_val_auc", float(lgb_auc))
        
        # Train XGBoost
        print("\n" + "-" * 60)
        xgb_model, xgb_auc = train_xgboost_model(X_train, y_train, X_val, y_val)
        
        # Log XGBoost metrics
        mlflow.log_metric("xgb_val_auc", float(xgb_auc))
        
        # ===== ENSEMBLE =====
        print("\n" + "-" * 60)
        logger.info("🎭 Creating ensemble predictions...")
        
        # Get predictions from both models
        lgb_val_pred = lgb_model.predict(X_val)  # type: ignore[arg-type]
        xgb_val_pred = xgb_model.predict(xgb.DMatrix(X_val))
        
        # Simple average ensemble
        ensemble_pred = 0.5 * lgb_val_pred + 0.5 * xgb_val_pred  # type: ignore[operator]
        ensemble_auc = roc_auc_score(y_val, ensemble_pred)
        
        logger.info(f"✓ Ensemble AUC: {ensemble_auc:.4f}")
        mlflow.log_metric("ensemble_val_auc", float(ensemble_auc))
        
        # ===== FEATURE IMPORTANCE =====
        print("\n" + "-" * 60)
        logger.info("📊 Analyzing feature importance...")
        
        # Get LightGBM feature importance
        lgb_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': lgb_model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        # Show top 30 features
        print("\n🔝 Top 30 Most Important Features:")
        print("=" * 60)
        for idx, row in lgb_importance.head(30).iterrows():
            print(f"  {row['feature']:<40} {row['importance']:>12.2f}")
        
        # Count feature types
        feature_types = {
            'GEO': sum(1 for f in feature_names if f.startswith('GEO_')),
            'DOC': sum(1 for f in feature_names if f.startswith('DOC_')),
            'BUREAU': sum(1 for f in feature_names if f.startswith('BUREAU_') or f.startswith('BB_')),
            'PREV': sum(1 for f in feature_names if f.startswith('PREV_')),
            'ORIGINAL': len(feature_names) - sum([
                sum(1 for f in feature_names if f.startswith(prefix))
                for prefix in ['GEO_', 'DOC_', 'BUREAU_', 'BB_', 'PREV_']
            ])
        }
        
        print("\n📈 Feature Count by Type:")
        print("=" * 60)
        for feat_type, count in feature_types.items():
            print(f"  {feat_type:<20} {count:>5} features")
        
        # ===== SAVE MODELS =====
        print("\n" + "=" * 60)
        logger.info("💾 Saving models...")
        
        models_dir = Path(config.get('models_path', 'models'))
        models_dir.mkdir(exist_ok=True, parents=True)
        
        # Save LightGBM
        lgb_path = models_dir / 'lightgbm_phase2.txt'
        lgb_model.save_model(str(lgb_path))
        logger.info(f"✓ LightGBM saved to {lgb_path}")
        mlflow.log_artifact(str(lgb_path))
        
        # Save XGBoost
        xgb_path = models_dir / 'xgboost_phase2.json'
        xgb_model.save_model(str(xgb_path))
        logger.info(f"✓ XGBoost saved to {xgb_path}")
        mlflow.log_artifact(str(xgb_path))
        
        # Save feature names
        feature_path = models_dir / 'feature_names_phase2.txt'
        with open(feature_path, 'w') as f:
            f.write('\n'.join(feature_names))
        logger.info(f"✓ Feature names saved to {feature_path}")
        mlflow.log_artifact(str(feature_path))
        
        # Save feature importance
        importance_path = models_dir / 'feature_importance_phase2.csv'
        lgb_importance.to_csv(importance_path, index=False)
        logger.info(f"✓ Feature importance saved to {importance_path}")
        mlflow.log_artifact(str(importance_path))
    
    # ===== SUMMARY =====
    print("\n" + "=" * 60)
    print("✅ PHASE 2 COMPLETE!")
    print("=" * 60)
    
    print("\n📊 Performance Summary:")
    print(f"  {'Baseline (Phase 1)':<25} {'AUC: 0.7385':>20}")
    print(f"  {'LightGBM (Phase 2)':<25} {f'AUC: {lgb_auc:.4f}':>20}")
    print(f"  {'XGBoost (Phase 2)':<25} {f'AUC: {xgb_auc:.4f}':>20}")
    print(f"  {'Ensemble (Phase 2)':<25} {f'AUC: {ensemble_auc:.4f}':>20}")
    
    improvement = ensemble_auc - 0.7385
    improvement_pct = (improvement / 0.7385) * 100
    
    print(f"\n🎯 Improvement: +{improvement:.4f} (+{improvement_pct:.2f}%)")
    
    print("\n💡 Next Steps:")
    print("  1. Run: mlflow ui --host 0.0.0.0")
    print("  2. View detailed metrics at http://localhost:5000")
    print("  3. Continue to Phase 3: Advanced Ensemble & Hyperparameter Tuning")
    print("  4. Analyze feature importance and SHAP values")
    
    print("\n" + "=" * 60)
    
    return {
        'lgb_auc': lgb_auc,
        'xgb_auc': xgb_auc,
        'ensemble_auc': ensemble_auc,
        'n_features': len(feature_names),
        'feature_types': feature_types
    }


if __name__ == "__main__":
    results = main()