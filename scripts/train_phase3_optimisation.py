""""""

Phase 3 COMPLETE Optimization: Hyperparameter Tuning + Advanced FeaturesPhase 3 Enhanced: Aggressive Optimization with Feature Selection



This script optimizes the complete 522-feature set from Phase 2:This script uses more aggressive strategies to push beyond Phase 2 performance:

✅ ALL 522 features (no selection - tree models handle dimensionality)- Wider hyperparameter search spaces

✅ Focused hyperparameter search (proven ranges)- Feature importance-based selection

✅ Advanced interaction features- Multiple ensemble strategies

✅ Stacking ensemble with 3 models- Calibration techniques

✅ Proper class weighting for imbalanced data"""



Target: 0.80+ AUC (currently at 0.7885)import sys

Expected gain: +0.015-0.030 AUC → 0.803-0.815import os

"""sys.path.insert(0, os.path.abspath('.'))



import sysimport pandas as pd

import osimport numpy as np

sys.path.insert(0, os.path.abspath('.'))import logging

from pathlib import Path

import pandas as pdimport joblib

import numpy as npfrom datetime import datetime

import loggingimport warnings

from pathlib import Pathwarnings.filterwarnings('ignore')

import joblib

from datetime import datetime# ML libraries

import warningsfrom sklearn.model_selection import train_test_split, StratifiedKFold

warnings.filterwarnings('ignore')from sklearn.metrics import roc_auc_score

from sklearn.feature_selection import SelectFromModel

# ML librariesfrom sklearn.calibration import CalibratedClassifierCV

from sklearn.model_selection import train_test_split, StratifiedKFoldfrom sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_scorefrom sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegressionimport lightgbm as lgb

import lightgbm as lgbimport xgboost as xgb

import xgboost as xgbimport optuna

try:from optuna.samplers import TPESampler

    import catboost as cbimport mlflow

    CATBOOST_AVAILABLE = True

except ImportError:# Our modules

    CATBOOST_AVAILABLE = Falsefrom src.credscope.utils.config import load_config

    print("⚠️  CatBoost not available - will use 2-model ensemble only")

# Configure logging

import optunalogging.basicConfig(

from optuna.samplers import TPESampler    level=logging.INFO,

import mlflow    format='%(levelname)s:%(name)s:%(message)s'

)

# Our moduleslogger = logging.getLogger(__name__)

from src.credscope.utils.config import load_config



# Configure loggingdef load_cached_features(config: dict) -> tuple:

logging.basicConfig(    """Load cached features from Phase 2"""

    level=logging.INFO,    feature_cache = Path(config['data']['features_path']) / 'engineered_features.pkl'

    format='%(levelname)s:%(name)s:%(message)s'    

)    if not feature_cache.exists():

logger = logging.getLogger(__name__)        raise FileNotFoundError(f"Cached features not found: {feature_cache}")

    

    logger.info(f"Loading cached features from {feature_cache}")

def load_cached_features_complete(config: dict) -> tuple:    cached = joblib.load(feature_cache)

    """Load cached complete features from Phase 2"""    

    feature_cache = Path(config['data']['features_path']) / 'engineered_features_COMPLETE.pkl'    return cached['train'], cached['test'], cached['features']

    

    if not feature_cache.exists():

        raise FileNotFoundError(def select_top_features(X_train, y_train, X_val, n_features: int = 200):

            f"Cached features not found: {feature_cache}\n"    """Select top N most important features using LightGBM

            f"Please run scripts/train_phase2_complete.py first!"    

        )    Args:

            X_train, y_train: Training data

    logger.info(f"Loading cached complete features from {feature_cache}")        X_val: Validation data

    cached = joblib.load(feature_cache)        n_features: Number of features to keep

            

    logger.info(f"✓ Loaded {len(cached['features'])} features")    Returns:

            Selected feature names, X_train_selected, X_val_selected

    return cached['train'], cached['test'], cached['features']    """

    logger.info(f"Selecting top {n_features} features using feature importance...")

    

def create_interaction_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:    # Train quick LightGBM to get feature importance

    """Create high-value interaction features based on Phase 2 feature importance    quick_model = lgb.LGBMClassifier(

            n_estimators=100,

    Top interactions identified:        num_leaves=31,

    1. EXT_SOURCE combinations        random_state=42,

    2. Payment behavior × Credit history        n_jobs=-1

    3. Credit utilization × Debt ratios    )

    4. Age/Employment × Income    quick_model.fit(X_train, y_train)

        

    Args:    # Get feature importance

        df: DataFrame with all Phase 2 features    importance = pd.DataFrame({

        is_train: Whether this is training data        'feature': X_train.columns,

                'importance': quick_model.feature_importances_

    Returns:    }).sort_values('importance', ascending=False)

        DataFrame with additional interaction features    

    """    # Select top features

    logger.info("Creating 20 high-value interaction features...")    top_features = importance.head(n_features)['feature'].tolist()

        

    interactions = {}    logger.info(f"✓ Selected {len(top_features)} features")

        logger.info(f"  Top 5: {top_features[:5]}")

    # 1. EXT_SOURCE INTERACTIONS (top 3 features)    

    if 'EXT_SOURCE_2' in df.columns and 'EXT_SOURCE_3' in df.columns:    return top_features, X_train[top_features], X_val[top_features]

        interactions['EXT_2x3'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']

        interactions['EXT_2_3_AVG'] = (df['EXT_SOURCE_2'] + df['EXT_SOURCE_3']) / 2

        interactions['EXT_2_3_DIFF'] = df['EXT_SOURCE_2'] - df['EXT_SOURCE_3']def optimize_lightgbm_aggressive(X_train, y_train, n_trials: int = 150):

        """Aggressive LightGBM optimization with wider search space"""

    if 'EXT_SOURCE_1' in df.columns and 'EXT_SOURCE_2' in df.columns:    

        interactions['EXT_1x2'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']    logger.info("🔥 Starting AGGRESSIVE LightGBM optimization...")

        

    if all(col in df.columns for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):    def objective(trial):

        interactions['EXT_ALL_AVG'] = (df['EXT_SOURCE_1'] + df['EXT_SOURCE_2'] + df['EXT_SOURCE_3']) / 3        # WIDER search space than before

        interactions['EXT_ALL_MIN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)        params = {

                'objective': 'binary',

    # 2. PAYMENT BEHAVIOR × CREDIT HISTORY            'metric': 'auc',

    if 'INST_RECENT_INST_PAYMENT_RATIO' in df.columns and 'BUREAU_DEBT_CREDIT_RATIO' in df.columns:            'verbosity': -1,

        interactions['PAYMENT_DEBT_INTERACTION'] = (            'random_state': 42,

            df['INST_RECENT_INST_PAYMENT_RATIO'] * df['BUREAU_DEBT_CREDIT_RATIO']            'force_col_wise': True,

        )            

                # More aggressive tree structure

    if 'INST_RECENT_INST_PAID_LATE' in df.columns and 'BUREAU_DAYS_CREDIT_MAX' in df.columns:            'num_leaves': trial.suggest_int('num_leaves', 10, 200),  # Wider range

        interactions['LATE_PAYMENT_HISTORY'] = (            'max_depth': trial.suggest_int('max_depth', 2, 15),      # Deeper trees

            df['INST_RECENT_INST_PAID_LATE'] * (df['BUREAU_DAYS_CREDIT_MAX'] / -365)            'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),  # More flexible

        )            'min_child_weight': trial.suggest_float('min_child_weight', 1e-4, 100.0, log=True),

                

    # 3. CREDIT UTILIZATION × DEBT            # Learning rate variations

    if 'CC_VERY_RECENT_CC_UTILIZATION' in df.columns and 'BUREAU_DEBT_CREDIT_RATIO' in df.columns:            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),  # Wider

        interactions['TOTAL_CREDIT_PRESSURE'] = (            'n_estimators': trial.suggest_int('n_estimators', 200, 3000),  # More trees

            df['CC_VERY_RECENT_CC_UTILIZATION'] * df['BUREAU_DEBT_CREDIT_RATIO']            

        )            # Aggressive sampling

                'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 1.0),  # More variation

    if 'CC_DRAWING_TO_PAYMENT_RATIO' in df.columns and 'AMT_CREDIT' in df.columns:            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.3, 1.0),

        interactions['DRAWING_CREDIT_RATIO'] = df['CC_DRAWING_TO_PAYMENT_RATIO'] * df['AMT_CREDIT']            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),

                

    # 4. AGE/EMPLOYMENT × INCOME            # Stronger regularization options

    if 'DAYS_BIRTH' in df.columns and 'AMT_CREDIT' in df.columns:            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 100.0, log=True),

        age_years = (-df['DAYS_BIRTH'] / 365)            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 100.0, log=True),

        interactions['AGE_CREDIT_INTERACTION'] = age_years * df['AMT_CREDIT']            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0, 20),

        interactions['CREDIT_PER_AGE'] = df['AMT_CREDIT'] / (age_years + 1)            

                # Additional parameters

    if 'DAYS_EMPLOYED' in df.columns and 'AMT_ANNUITY' in df.columns:            'path_smooth': trial.suggest_float('path_smooth', 0, 1),

        employment_years = np.maximum(-df['DAYS_EMPLOYED'] / 365, 0)            'max_bin': trial.suggest_int('max_bin', 100, 500),

        interactions['EMPLOYMENT_INCOME'] = employment_years * df['AMT_ANNUITY']        }

            

    # 5. PREVIOUS APPLICATION × REFUSAL PATTERNS        # 3-fold CV (faster than 5-fold)

    if 'PREV_APPROVAL_RATE' in df.columns and 'PREV_APP_CREDIT_RATIO' in df.columns:        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        interactions['APPROVAL_CREDIT_INTERACTION'] = (        scores = []

            df['PREV_APPROVAL_RATE'] * df['PREV_APP_CREDIT_RATIO']        

        )        for train_idx, val_idx in skf.split(X_train, y_train):

                X_tr = X_train.iloc[train_idx] if isinstance(X_train, pd.DataFrame) else X_train[train_idx]

    if 'PREV_REFUSAL_RATE' in df.columns and 'BUREAU_DAYS_CREDIT_MAX' in df.columns:            X_va = X_train.iloc[val_idx] if isinstance(X_train, pd.DataFrame) else X_train[val_idx]

        interactions['REFUSAL_CREDIT_AGE'] = (            y_tr = y_train.iloc[train_idx] if isinstance(y_train, pd.Series) else y_train[train_idx]

            df['PREV_REFUSAL_RATE'] * (df['BUREAU_DAYS_CREDIT_MAX'] / -365)            y_va = y_train.iloc[val_idx] if isinstance(y_train, pd.Series) else y_train[val_idx]

        )            

                model = lgb.LGBMClassifier(**params)

    # 6. POS/CASH × INSTALLMENTS            model.fit(

    if 'POS_AVG_COMPLETION' in df.columns and 'INST_PAYMENT_COMPLETION_RATE' in df.columns:                X_tr, y_tr,

        interactions['OVERALL_COMPLETION'] = (                eval_set=[(X_va, y_va)],

            df['POS_AVG_COMPLETION'] + df['INST_PAYMENT_COMPLETION_RATE']                callbacks=[

        ) / 2                    lgb.early_stopping(50, verbose=False),

                        lgb.log_evaluation(0)

    # 7. DOCUMENT QUALITY × CREDIT                ]

    if 'DOC_PER_CREDIT_UNIT' in df.columns and 'AMT_CREDIT' in df.columns:            )

        interactions['DOC_CREDIT_QUALITY'] = df['DOC_PER_CREDIT_UNIT'] * df['AMT_CREDIT']            

                y_pred = model.predict_proba(X_va)[:, 1]  # type: ignore

    # 8. GEOGRAPHIC × INCOME            score = roc_auc_score(y_va, y_pred)

    if 'GEO_REGION_POPULATION_RELATIVE_LOG' in df.columns and 'AMT_ANNUITY' in df.columns:            scores.append(score)

        interactions['REGION_INCOME'] = df['GEO_REGION_POPULATION_RELATIVE_LOG'] * df['AMT_ANNUITY']        

            return float(np.mean(scores))

    # 9. RECENT CREDIT CARD × INSTALLMENTS    

    if 'CC_VERY_RECENT_CC_UTILIZATION' in df.columns and 'INST_RECENT_INST_DPD' in df.columns:    # Create study with more aggressive sampling

        interactions['RECENT_RISK_SIGNAL'] = (    study = optuna.create_study(

            df['CC_VERY_RECENT_CC_UTILIZATION'] * (df['INST_RECENT_INST_DPD'] + 1)        direction='maximize',

        )        sampler=TPESampler(seed=42, n_startup_trials=20)  # More exploration

        )

    # 10. GOODS PRICE × INCOME RATIO    

    if 'AMT_GOODS_PRICE' in df.columns and 'AMT_ANNUITY' in df.columns:    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        interactions['GOODS_TO_INCOME'] = df['AMT_GOODS_PRICE'] / (df['AMT_ANNUITY'] * 12 + 1)    

        logger.info(f"✓ Best LightGBM AUC: {study.best_value:.4f}")

    # Convert to DataFrame and add to original    logger.info(f"  Best params: {study.best_params}")

    interaction_df = pd.DataFrame(interactions)    

        return study.best_params, study.best_value

    # Handle infinite/nan values

    interaction_df = interaction_df.replace([np.inf, -np.inf], np.nan)

    for col in interaction_df.columns:def optimize_xgboost_aggressive(X_train, y_train, n_trials: int = 150):

        interaction_df[col] = interaction_df[col].fillna(interaction_df[col].median())    """Aggressive XGBoost optimization with wider search space"""

        

    result = pd.concat([df, interaction_df], axis=1)    logger.info("🔥 Starting AGGRESSIVE XGBoost optimization...")

        

    logger.info(f"✓ Created {len(interactions)} interaction features")    def objective(trial):

    logger.info(f"  New shape: {result.shape}")        # WIDER search space

            params = {

    return result            'objective': 'binary:logistic',

            'eval_metric': 'auc',

            'tree_method': 'hist',

def optimize_lightgbm(trial, X_train, y_train, X_val, y_val):            'random_state': 42,

    """Optimize LightGBM with FOCUSED hyperparameter ranges            'verbosity': 0,

                

    Key improvements from Phase 2:            # More aggressive tree structure

    - Narrower ranges based on proven values            'max_depth': trial.suggest_int('max_depth', 2, 12),  # Deeper

    - class_weight='balanced' for imbalanced data            'min_child_weight': trial.suggest_int('min_child_weight', 1, 500),  # Wider

    - Better regularization range            'gamma': trial.suggest_float('gamma', 1e-8, 10.0, log=True),

    """            

    params = {            # Learning variations

        'objective': 'binary',            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),

        'metric': 'auc',            'n_estimators': trial.suggest_int('n_estimators', 200, 3000),

        'verbosity': -1,            

        'random_state': 42,            # Aggressive sampling

        'force_col_wise': True,            'subsample': trial.suggest_float('subsample', 0.3, 1.0),

        'class_weight': 'balanced',  # NEW: Handle imbalanced data            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),

                    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 1.0),

        # Narrowed tree structure (focus around proven values)            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.3, 1.0),

        'num_leaves': trial.suggest_int('num_leaves', 50, 100),            

        'max_depth': trial.suggest_int('max_depth', 5, 9),            # Stronger regularization

        'min_child_samples': trial.suggest_int('min_child_samples', 20, 80),            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100.0, log=True),

        'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 10.0, log=True),            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100.0, log=True),

                    

        # Better learning rate range            # Additional parameters

        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),            'max_bin': trial.suggest_int('max_bin', 100, 500),

        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 20),  # Handle imbalance

                }

        # Reasonable sampling (less aggressive)        

        'feature_fraction': trial.suggest_float('feature_fraction', 0.65, 0.95),        # 3-fold CV

        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.65, 0.95),        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        'bagging_freq': trial.suggest_int('bagging_freq', 3, 7),        scores = []

                

        # Focused regularization        for train_idx, val_idx in skf.split(X_train, y_train):

        'lambda_l1': trial.suggest_float('lambda_l1', 0.001, 5.0, log=True),            X_tr = X_train.iloc[train_idx] if isinstance(X_train, pd.DataFrame) else X_train[train_idx]

        'lambda_l2': trial.suggest_float('lambda_l2', 0.001, 5.0, log=True),            X_va = X_train.iloc[val_idx] if isinstance(X_train, pd.DataFrame) else X_train[val_idx]

        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0, 5),            y_tr = y_train.iloc[train_idx] if isinstance(y_train, pd.Series) else y_train[train_idx]

                    y_va = y_train.iloc[val_idx] if isinstance(y_train, pd.Series) else y_train[val_idx]

        # Additional parameters            

        'path_smooth': trial.suggest_float('path_smooth', 0, 0.5),            model = xgb.XGBClassifier(**params)

        'max_bin': trial.suggest_int('max_bin', 200, 300),            model.fit(

    }                X_tr, y_tr,

                    eval_set=[(X_va, y_va)],

    # Train with early stopping                early_stopping_rounds=50,

    model = lgb.LGBMClassifier(**params)                verbose=False

    model.fit(            )

        X_train, y_train,            

        eval_set=[(X_val, y_val)],            y_pred = model.predict_proba(X_va)[:, 1]  # type: ignore

        eval_metric='auc',            score = roc_auc_score(y_va, y_pred)

        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]            scores.append(score)

    )        

            return float(np.mean(scores))

    # Predict and evaluate    

    y_pred = model.predict_proba(X_val)[:, 1]  # type: ignore[arg-type]    study = optuna.create_study(

    auc = roc_auc_score(y_val, y_pred)        direction='maximize',

            sampler=TPESampler(seed=42, n_startup_trials=20)

    return auc    )

    

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

def optimize_xgboost(trial, X_train, y_train, X_val, y_val):    

    """Optimize XGBoost with FOCUSED hyperparameter ranges"""    logger.info(f"✓ Best XGBoost AUC: {study.best_value:.4f}")

        logger.info(f"  Best params: {study.best_params}")

    # Calculate scale_pos_weight for imbalanced data    

    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()    return study.best_params, study.best_value

    

    params = {

        'objective': 'binary:logistic',def create_calibrated_models(lgb_model, xgb_model, X_train, y_train):

        'eval_metric': 'auc',    """Create probability-calibrated versions of models"""

        'tree_method': 'hist',    

        'random_state': 42,    logger.info("🎯 Creating calibrated models...")

        'verbosity': 0,    

        'scale_pos_weight': scale_pos_weight,  # NEW: Handle imbalanced data    # Calibrate LightGBM

            lgb_calibrated = CalibratedClassifierCV(

        # Narrowed tree structure        lgb_model,

        'max_depth': trial.suggest_int('max_depth', 4, 8),        method='isotonic',

        'min_child_weight': trial.suggest_int('min_child_weight', 10, 100),        cv=3

        'gamma': trial.suggest_float('gamma', 0.001, 2.0, log=True),    )

            lgb_calibrated.fit(X_train, y_train)

        # Better learning range    

        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),    # Calibrate XGBoost  

        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),    xgb_calibrated = CalibratedClassifierCV(

                xgb_model,

        # Reasonable sampling        method='isotonic',

        'subsample': trial.suggest_float('subsample', 0.65, 0.95),        cv=3

        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 0.95),    )

        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.65, 0.95),    xgb_calibrated.fit(X_train, y_train)

        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.65, 0.95),    

            logger.info("✓ Calibration complete")

        # Focused regularization    

        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 5.0, log=True),    return lgb_calibrated, xgb_calibrated

        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 5.0, log=True),

        

        # Additional parametersdef create_voting_ensemble(lgb_model, xgb_model, X_train, y_train, X_val, y_val):

        'max_delta_step': trial.suggest_int('max_delta_step', 0, 5),    """Create sklearn VotingClassifier ensemble"""

    }    

        logger.info("🗳️ Creating voting ensemble...")

    # Train with early stopping    

    model = xgb.XGBClassifier(**params)    # Try different weight combinations

    model.fit(    best_auc = 0

        X_train, y_train,    best_weights = None

        eval_set=[(X_val, y_val)],    

        verbose=False    for w1 in [0.3, 0.4, 0.5, 0.6, 0.7]:

    )        w2 = 1 - w1

            

    # Predict and evaluate        voting = VotingClassifier(

    y_pred = model.predict_proba(X_val)[:, 1]  # type: ignore[arg-type]            estimators=[('lgb', lgb_model), ('xgb', xgb_model)],

    auc = roc_auc_score(y_val, y_pred)            voting='soft',

                weights=[w1, w2]

    return auc        )

        voting.fit(X_train, y_train)

        

def optimize_catboost(trial, X_train, y_train, X_val, y_val):        y_pred = voting.predict_proba(X_val)[:, 1]

    """Optimize CatBoost with FOCUSED hyperparameter ranges"""        auc = roc_auc_score(y_val, y_pred)

            

    params = {        if auc > best_auc:

        'loss_function': 'Logloss',            best_auc = auc

        'eval_metric': 'AUC',            best_weights = [w1, w2]

        'random_seed': 42,    

        'verbose': 0,    logger.info(f"✓ Best voting ensemble: {best_weights} → AUC: {best_auc:.4f}")

        'auto_class_weights': 'Balanced',  # NEW: Handle imbalanced data    

            # Create final voting ensemble

        # Tree structure    final_voting = VotingClassifier(

        'depth': trial.suggest_int('depth', 4, 8),        estimators=[('lgb', lgb_model), ('xgb', xgb_model)],

        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 80),        voting='soft',

                weights=best_weights

        # Learning rate    )

        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),    final_voting.fit(X_train, y_train)

        'iterations': trial.suggest_int('iterations', 500, 2000),    

            return final_voting, best_auc

        # Regularization

        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),

        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),def create_blending_ensemble(models_dict, X_val, y_val):

        'random_strength': trial.suggest_float('random_strength', 0, 1),    """Create blended predictions with optimized weights"""

            

        # Sampling    logger.info("🎨 Creating blended ensemble...")

        'subsample': trial.suggest_float('subsample', 0.65, 0.95),    

    }    # Get predictions from all models

        predictions = {}

    # Train with early stopping    for name, model in models_dict.items():

    model = cb.CatBoostClassifier(**params)        if hasattr(model, 'predict_proba'):

    model.fit(            pred = model.predict_proba(X_val)[:, 1]

        X_train, y_train,        else:

        eval_set=(X_val, y_val),            pred = model.predict(X_val)

        early_stopping_rounds=100,        predictions[name] = pred

        verbose=False    

    )    # Optimize weights with Optuna

        def objective(trial):

    # Predict and evaluate        weights = []

    y_pred = model.predict_proba(X_val)[:, 1]        remaining = 1.0

    auc = roc_auc_score(y_val, y_pred)        

            for i in range(len(models_dict) - 1):

    return auc            w = trial.suggest_float(f'w{i}', 0.0, remaining)

            weights.append(w)

            remaining -= w

def main():        weights.append(remaining)

    """Main training pipeline for Phase 3 Complete Optimization"""        

            # Weighted average

    print("=" * 70)        blend = sum(w * predictions[name] for w, name in zip(weights, predictions.keys()))

    print("🔥 CredScope Phase 3 COMPLETE: Optimization + Advanced Features")        return float(roc_auc_score(y_val, blend))

    print("=" * 70)    

    print("\n⭐ Strategy:")    study = optuna.create_study(direction='maximize')

    print("  ✅ Use ALL 522 features (no selection)")    study.optimize(objective, n_trials=100, show_progress_bar=False)

    print("  ✅ Add 20 interaction features → 542 total")    

    print("  ✅ Focused hyperparameter optimization (150 trials)")    # Get best weights

    print("  ✅ 3-model stacking ensemble (LightGBM + XGBoost + CatBoost)")    best_weights = []

    print("  ✅ Proper class weighting for imbalanced data")    remaining = 1.0

    print("\n🎯 Current AUC: 0.7885")    for i in range(len(models_dict) - 1):

    print("🎯 Target AUC: 0.8000+")        w = study.best_params[f'w{i}']

    print("🎯 Expected: 0.803-0.815 AUC")        best_weights.append(w)

    print("\n⏱️  Expected Duration: 2-3 hours\n")        remaining -= w

        best_weights.append(remaining)

    # Load config    

    config = load_config('config.yaml')    # Create final blend

        final_blend = sum(w * predictions[name] for w, name in zip(best_weights, predictions.keys()))

    # Setup MLflow    blend_auc = roc_auc_score(y_val, final_blend)

    mlflow.set_experiment("credscope_phase3_complete")    

        logger.info(f"✓ Blended ensemble AUC: {blend_auc:.4f}")

    with mlflow.start_run(run_name=f"phase3_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):    logger.info(f"  Weights: {dict(zip(predictions.keys(), best_weights))}")

            

        # ===== LOAD CACHED FEATURES =====    return final_blend, blend_auc, dict(zip(predictions.keys(), best_weights))

        print("\n" + "=" * 70)

        print("📁 Step 1/7: Loading Phase 2 Complete Features (522 features)")

        print("=" * 70)def main():

            """Main enhanced optimization pipeline"""

        train_df, test_df, feature_names = load_cached_features_complete(config)    

            print("=" * 70)

        logger.info(f"✓ Loaded features: {len(feature_names)}")    print("🔥 CredScope Phase 3 ENHANCED: Aggressive Optimization")

        logger.info(f"✓ Train shape: {train_df.shape}")    print("=" * 70)

        logger.info(f"✓ Test shape: {test_df.shape}")    print("\n💡 Strategy:")

            print("  ✓ Feature selection (top 200 features)")

        mlflow.log_param("base_features", len(feature_names))    print("  ✓ Wider hyperparameter search spaces")

            print("  ✓ Probability calibration")

        # ===== CREATE INTERACTION FEATURES =====    print("  ✓ Multiple ensemble strategies")

        print("\n" + "=" * 70)    print("  ✓ 3-fold CV (faster, more trials)")

        print("🧠 Step 2/7: Creating Advanced Interaction Features")    print("\n⏱️  Expected Duration: 30-45 minutes\n")

        print("=" * 70)    

            # Load config

        target = train_df['TARGET'].copy()    config = load_config()

        train_df = train_df.drop(columns=['TARGET'])    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])

            mlflow.set_experiment("credscope_phase3_enhanced")

        train_df = create_interaction_features(train_df, is_train=True)    

        test_df = create_interaction_features(test_df, is_train=False)    # Load features

            print("=" * 70)

        # Update feature names    print("📁 Loading Phase 2 features")

        all_feature_names = [col for col in train_df.columns]    print("=" * 70)

        n_interactions = len(all_feature_names) - len(feature_names)    

            train_df, test_df, feature_names = load_cached_features(config)

        logger.info(f"✓ Total features: {len(all_feature_names)} (base: {len(feature_names)}, interactions: {n_interactions})")    

            X = train_df.drop(columns=['TARGET'])

        mlflow.log_param("total_features", len(all_feature_names))    y = train_df['TARGET']

        mlflow.log_param("interaction_features", n_interactions)    

            X_train, X_val, y_train, y_val = train_test_split(

        # ===== PREPARE DATA =====        X, y, test_size=0.2, random_state=42, stratify=y

        print("\n" + "=" * 70)    )

        print("📊 Step 3/7: Preparing Train/Validation Split")    

        print("=" * 70)    logger.info(f"✓ Loaded data: {X_train.shape[0]:,} train, {X_val.shape[0]:,} val")

            

        X_train, X_val, y_train, y_val = train_test_split(    with mlflow.start_run(run_name=f"enhanced_{datetime.now().strftime('%Y%m%d_%H%M')}"):

            train_df, target, test_size=0.2, random_state=42, stratify=target        

        )        # ===== FEATURE SELECTION =====

                print("\n" + "=" * 70)

        logger.info(f"✓ Train: {X_train.shape[0]} samples")        print("🎯 Step 1/5: Feature Selection")

        logger.info(f"✓ Validation: {X_val.shape[0]} samples")        print("=" * 70)

        logger.info(f"✓ Features: {X_train.shape[1]}")        

        logger.info(f"✓ Default rate (train): {y_train.mean():.3%}")        top_features, X_train_selected, X_val_selected = select_top_features(

        logger.info(f"✓ Default rate (val): {y_val.mean():.3%}")            X_train, y_train, X_val, n_features=200

                )

        mlflow.log_param("train_samples", X_train.shape[0])        

        mlflow.log_param("val_samples", X_val.shape[0])        mlflow.log_param("n_features_selected", len(top_features))

                

        # ===== OPTIMIZE LIGHTGBM =====        # ===== OPTIMIZE LIGHTGBM =====

        print("\n" + "=" * 70)        print("\n" + "=" * 70)

        print("🚀 Step 4/7: Optimizing LightGBM (150 trials)")        print("🔥 Step 2/5: Aggressive LightGBM Optimization (150 trials)")

        print("=" * 70)        print("=" * 70)

                

        study_lgb = optuna.create_study(        lgb_params, lgb_cv_score = optimize_lightgbm_aggressive(

            direction='maximize',            X_train_selected, y_train, n_trials=150

            sampler=TPESampler(seed=42),        )

            study_name='lgb_optimization'        

        )        # Train final LightGBM

                lgb_final = lgb.LGBMClassifier(**lgb_params)

        study_lgb.optimize(        lgb_final.fit(X_train_selected, y_train)

            lambda trial: optimize_lightgbm(trial, X_train, y_train, X_val, y_val),        lgb_val_pred = lgb_final.predict_proba(X_val_selected)[:, 1]  # type: ignore

            n_trials=150,        lgb_val_auc = roc_auc_score(y_val, lgb_val_pred)

            show_progress_bar=True,        

            callbacks=[        logger.info(f"✓ LightGBM Val AUC: {lgb_val_auc:.4f}")

                lambda study, trial: print(f"Trial {trial.number}: AUC = {trial.value:.6f}")        mlflow.log_metric("lgb_val_auc", float(lgb_val_auc))

            ]        

        )        # ===== OPTIMIZE XGBOOST =====

                print("\n" + "=" * 70)

        best_lgb_params = study_lgb.best_params        print("🔥 Step 3/5: Aggressive XGBoost Optimization (150 trials)")

        best_lgb_auc = study_lgb.best_value        print("=" * 70)

                

        logger.info(f"✓ Best LightGBM AUC: {best_lgb_auc:.6f}")        xgb_params, xgb_cv_score = optimize_xgboost_aggressive(

        logger.info(f"✓ Best params: {best_lgb_params}")            X_train_selected, y_train, n_trials=150

                )

        mlflow.log_metric("best_lgb_auc", best_lgb_auc)        

        mlflow.log_params({f"lgb_{k}": v for k, v in best_lgb_params.items()})        # Train final XGBoost

                xgb_final = xgb.XGBClassifier(**xgb_params)

        # Train final LightGBM        xgb_final.fit(X_train_selected, y_train)

        final_lgb_params = {        xgb_val_pred = xgb_final.predict_proba(X_val_selected)[:, 1]  # type: ignore

            'objective': 'binary',        xgb_val_auc = roc_auc_score(y_val, xgb_val_pred)

            'metric': 'auc',        

            'verbosity': -1,        logger.info(f"✓ XGBoost Val AUC: {xgb_val_auc:.4f}")

            'random_state': 42,        mlflow.log_metric("xgb_val_auc", float(xgb_val_auc))

            'force_col_wise': True,        

            'class_weight': 'balanced',        # ===== CALIBRATION =====

            **best_lgb_params        print("\n" + "=" * 70)

        }        print("🎯 Step 4/5: Probability Calibration")

                print("=" * 70)

        lgb_model = lgb.LGBMClassifier(**final_lgb_params)        

        lgb_model.fit(        lgb_calibrated, xgb_calibrated = create_calibrated_models(

            X_train, y_train,            lgb_final, xgb_final, X_train_selected, y_train

            eval_set=[(X_val, y_val)],        )

            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]        

        )        lgb_cal_pred = lgb_calibrated.predict_proba(X_val_selected)[:, 1]

                xgb_cal_pred = xgb_calibrated.predict_proba(X_val_selected)[:, 1]

        lgb_val_pred = lgb_model.predict_proba(X_val)[:, 1]  # type: ignore[arg-type]        

        lgb_val_auc = roc_auc_score(y_val, lgb_val_pred)        lgb_cal_auc = roc_auc_score(y_val, lgb_cal_pred)

                xgb_cal_auc = roc_auc_score(y_val, xgb_cal_pred)

        logger.info(f"✓ Final LightGBM validation AUC: {lgb_val_auc:.6f}")        

                logger.info(f"✓ Calibrated LightGBM AUC: {lgb_cal_auc:.4f}")

        # ===== OPTIMIZE XGBOOST =====        logger.info(f"✓ Calibrated XGBoost AUC: {xgb_cal_auc:.4f}")

        print("\n" + "=" * 70)        mlflow.log_metric("lgb_calibrated_auc", float(lgb_cal_auc))

        print("🚀 Step 5/7: Optimizing XGBoost (150 trials)")        mlflow.log_metric("xgb_calibrated_auc", float(xgb_cal_auc))

        print("=" * 70)        

                # ===== ENSEMBLE STRATEGIES =====

        study_xgb = optuna.create_study(        print("\n" + "=" * 70)

            direction='maximize',        print("🎭 Step 5/5: Multiple Ensemble Strategies")

            sampler=TPESampler(seed=42),        print("=" * 70)

            study_name='xgb_optimization'        

        )        # Strategy 1: Voting Ensemble

                voting, voting_auc = create_voting_ensemble(

        study_xgb.optimize(            lgb_final, xgb_final, X_train_selected, y_train, X_val_selected, y_val

            lambda trial: optimize_xgboost(trial, X_train, y_train, X_val, y_val),        )

            n_trials=150,        mlflow.log_metric("voting_auc", float(voting_auc))

            show_progress_bar=True,        

            callbacks=[        # Strategy 2: Blended Ensemble (all models)

                lambda study, trial: print(f"Trial {trial.number}: AUC = {trial.value:.6f}")        all_models = {

            ]            'lgb': lgb_final,

        )            'xgb': xgb_final,

                    'lgb_cal': lgb_calibrated,

        best_xgb_params = study_xgb.best_params            'xgb_cal': xgb_calibrated

        best_xgb_auc = study_xgb.best_value        }

                

        logger.info(f"✓ Best XGBoost AUC: {best_xgb_auc:.6f}")        blend_pred, blend_auc, blend_weights = create_blending_ensemble(

        logger.info(f"✓ Best params: {best_xgb_params}")            all_models, X_val_selected, y_val

                )

        mlflow.log_metric("best_xgb_auc", best_xgb_auc)        mlflow.log_metric("blend_auc", float(blend_auc))

        mlflow.log_params({f"xgb_{k}": v for k, v in best_xgb_params.items()})        

                # Strategy 3: Simple Average

        # Calculate scale_pos_weight        simple_avg = (lgb_val_pred + xgb_val_pred) / 2

        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()        simple_avg_auc = roc_auc_score(y_val, simple_avg)

                mlflow.log_metric("simple_avg_auc", float(simple_avg_auc))

        # Train final XGBoost        

        final_xgb_params = {        # ===== RESULTS =====

            'objective': 'binary:logistic',        print("\n" + "=" * 70)

            'eval_metric': 'auc',        print("📊 ENHANCED PHASE 3 RESULTS")

            'tree_method': 'hist',        print("=" * 70)

            'random_state': 42,        

            'verbosity': 0,        results = {

            'scale_pos_weight': scale_pos_weight,            'Baseline (Phase 1)': 0.7385,

            **best_xgb_params            'Phase 2 - LightGBM': 0.7755,

        }            'Phase 2 - XGBoost': 0.7785,

                    'Phase 2 - Simple Ensemble': 0.7794,

        xgb_model = xgb.XGBClassifier(**final_xgb_params)            'Phase 3 - Optimized LightGBM': lgb_val_auc,

        xgb_model.fit(            'Phase 3 - Optimized XGBoost': xgb_val_auc,

            X_train, y_train,            'Phase 3 - Calibrated LightGBM': lgb_cal_auc,

            eval_set=[(X_val, y_val)],            'Phase 3 - Calibrated XGBoost': xgb_cal_auc,

            verbose=False            'Phase 3 - Voting Ensemble': voting_auc,

        )            'Phase 3 - Simple Average': simple_avg_auc,

                    'Phase 3 - Blended Ensemble': blend_auc,

        xgb_val_pred = xgb_model.predict_proba(X_val)[:, 1]  # type: ignore[arg-type]        }

        xgb_val_auc = roc_auc_score(y_val, xgb_val_pred)        

                print("\n🎯 Performance Comparison:")

        logger.info(f"✓ Final XGBoost validation AUC: {xgb_val_auc:.6f}")        print("-" * 70)

                for name, score in results.items():

        # ===== OPTIMIZE CATBOOST (if available) =====            improvement = ((score - 0.7385) / 0.7385) * 100

        catboost_available = CATBOOST_AVAILABLE            vs_phase2 = ((score - 0.7794) / 0.7794) * 100 if score > 0.7794 else 0

        cb_val_pred = None            print(f"  {name:<40} AUC: {score:.4f}  (+{improvement:>5.2f}%)")

        cb_val_auc = None            if vs_phase2 > 0:

                        print(f"  {'':<40}          vs Phase2: +{vs_phase2:.2f}%")

        if catboost_available:        

            print("\n" + "=" * 70)        # Best model

            print("🚀 Step 6/7: Optimizing CatBoost (150 trials)")        best_name = max(results.items(), key=lambda x: x[1])[0]

            print("=" * 70)        best_auc = max(results.values())

                    

            study_cb = optuna.create_study(        print(f"\n🏆 Best Model: {best_name}")

                direction='maximize',        print(f"   Final AUC: {best_auc:.4f}")

                sampler=TPESampler(seed=42),        print(f"   Improvement vs Baseline: +{best_auc - 0.7385:.4f} (+{((best_auc - 0.7385) / 0.7385) * 100:.2f}%)")

                study_name='cb_optimization'        print(f"   Improvement vs Phase 2: +{best_auc - 0.7794:.4f} (+{((best_auc - 0.7794) / 0.7794) * 100:.2f}%)")

            )        

                    # ===== SAVE MODELS =====

            study_cb.optimize(        print("\n" + "=" * 70)

                lambda trial: optimize_catboost(trial, X_train, y_train, X_val, y_val),        print("💾 Saving Models")

                n_trials=150,        print("=" * 70)

                show_progress_bar=True,        

                callbacks=[        models_dir = Path('models')

                    lambda study, trial: print(f"Trial {trial.number}: AUC = {trial.value:.6f}")        

                ]        # Save best individual models

            )        joblib.dump(lgb_calibrated, models_dir / 'lgb_calibrated_enhanced.pkl')

                    joblib.dump(xgb_calibrated, models_dir / 'xgb_calibrated_enhanced.pkl')

            best_cb_params = study_cb.best_params        joblib.dump(voting, models_dir / 'voting_ensemble_enhanced.pkl')

            best_cb_auc = study_cb.best_value        

                    # Save metadata

            logger.info(f"✓ Best CatBoost AUC: {best_cb_auc:.6f}")        metadata = {

            logger.info(f"✓ Best params: {best_cb_params}")            'best_lgb_params': lgb_params,

                        'best_xgb_params': xgb_params,

            mlflow.log_metric("best_cb_auc", best_cb_auc)            'blend_weights': blend_weights,

            mlflow.log_params({f"cb_{k}": v for k, v in best_cb_params.items()})            'selected_features': top_features,

                        'results': results,

            # Train final CatBoost            'best_model': best_name,

            final_cb_params = {            'best_auc': best_auc

                'loss_function': 'Logloss',        }

                'eval_metric': 'AUC',        joblib.dump(metadata, models_dir / 'phase3_enhanced_metadata.pkl')

                'random_seed': 42,        

                'verbose': 0,        # Save results CSV

                'auto_class_weights': 'Balanced',        results_df = pd.DataFrame({

                **best_cb_params            'Model': list(results.keys()),

            }            'AUC': list(results.values())

                    })

            cb_model = cb.CatBoostClassifier(**final_cb_params)        results_df.to_csv(models_dir / 'phase3_enhanced_results.csv', index=False)

            cb_model.fit(        

                X_train, y_train,        logger.info("✓ All models and metadata saved")

                eval_set=(X_val, y_val),        

                early_stopping_rounds=100,        print("\n" + "=" * 70)

                verbose=False        print("✅ ENHANCED PHASE 3 COMPLETE!")

            )        print("=" * 70)

                    

            cb_val_pred = cb_model.predict_proba(X_val)[:, 1]        if best_auc >= 0.82:

            cb_val_auc = roc_auc_score(y_val, cb_val_pred)            print("\n🎊 🏆 EXCELLENT! Target exceeded! (≥0.82 AUC)")

                    elif best_auc >= 0.80:

            logger.info(f"✓ Final CatBoost validation AUC: {cb_val_auc:.6f}")            print("\n🎯 GREAT! Strong improvement achieved!")

        else:        elif best_auc > 0.7794:

            print("\n" + "=" * 70)            print("\n📈 GOOD! We improved over Phase 2!")

            print("⚠️  Step 6/7: CatBoost not available - skipping")        else:

            print("=" * 70)            print("\n💡 Model may be at ceiling with current features")

                    print("   Consider: Feature engineering, external data, different algorithms")

        # ===== CREATE STACKING ENSEMBLE =====        

        print("\n" + "=" * 70)        print(f"\n💡 Next Steps:")

        print("🎭 Step 7/7: Creating Stacking Ensemble")        print(f"  1. Review feature importance of selected features")

        print("=" * 70)        print(f"  2. Analyze predictions for insights")

                print(f"  3. Consider additional feature engineering if needed")

        if catboost_available and cb_val_pred is not None:        print(f"  4. Move to deployment if satisfied with performance")

            # 3-model stacking        

            logger.info("Using 3-model stacking: LightGBM + XGBoost + CatBoost")        return {

                        'best_auc': best_auc,

            # Stack predictions            'improvement_vs_phase2': best_auc - 0.7794,

            stacked_train = np.column_stack([            'all_results': results

                lgb_val_pred,        }

                xgb_val_pred,

                cb_val_pred

            ])if __name__ == "__main__":

                results = main()
            # Train meta-learner (Logistic Regression)
            meta_model = LogisticRegression(random_state=42, max_iter=1000)
            meta_model.fit(stacked_train, y_val)
            
            # Get ensemble weights from meta-model
            weights = np.abs(meta_model.coef_[0])
            weights = weights / weights.sum()
            
            logger.info(f"✓ Learned ensemble weights:")
            logger.info(f"   LightGBM: {weights[0]:.3f}")
            logger.info(f"   XGBoost: {weights[1]:.3f}")
            logger.info(f"   CatBoost: {weights[2]:.3f}")
            
            # Weighted ensemble prediction
            ensemble_pred = (
                weights[0] * lgb_val_pred +
                weights[1] * xgb_val_pred +
                weights[2] * cb_val_pred
            )  # type: ignore[operator]
            
            mlflow.log_param("ensemble_type", "3-model-stacking")
            mlflow.log_param("lgb_weight", float(weights[0]))
            mlflow.log_param("xgb_weight", float(weights[1]))
            mlflow.log_param("cb_weight", float(weights[2]))
        else:
            # 2-model weighted average
            logger.info("Using 2-model weighted ensemble: LightGBM + XGBoost")
            
            # Optimize weights
            best_auc = 0
            best_w = 0.5
            
            for w in np.linspace(0.3, 0.7, 41):
                pred = w * xgb_val_pred + (1 - w) * lgb_val_pred  # type: ignore[operator]
                auc = roc_auc_score(y_val, pred)
                if auc > best_auc:
                    best_auc = auc
                    best_w = w
            
            logger.info(f"✓ Optimized weights:")
            logger.info(f"   XGBoost: {best_w:.3f}")
            logger.info(f"   LightGBM: {1-best_w:.3f}")
            
            ensemble_pred = best_w * xgb_val_pred + (1 - best_w) * lgb_val_pred  # type: ignore[operator]
            
            mlflow.log_param("ensemble_type", "2-model-weighted")
            mlflow.log_param("xgb_weight", float(best_w))
            mlflow.log_param("lgb_weight", float(1 - best_w))
        
        ensemble_auc = roc_auc_score(y_val, ensemble_pred)
        
        logger.info(f"\n✓ Ensemble validation AUC: {ensemble_auc:.6f}")
        
        mlflow.log_metric("ensemble_auc", ensemble_auc)
        
        # ===== FINAL RESULTS =====
        print("\n" + "=" * 70)
        print("📊 FINAL RESULTS")
        print("=" * 70)
        
        print("\n🎯 Model Performance:")
        print("-" * 70)
        print(f"  LightGBM               AUC: {lgb_val_auc:.6f}")
        print(f"  XGBoost                AUC: {xgb_val_auc:.6f}")
        if cb_val_auc is not None:
            print(f"  CatBoost               AUC: {cb_val_auc:.6f}")
        print(f"  Ensemble               AUC: {ensemble_auc:.6f}")
        
        print("\n📈 Improvement Tracking:")
        print("-" * 70)
        baseline_auc = 0.7385
        phase2_partial_auc = 0.7794
        phase2_complete_auc = 0.7885
        
        print(f"  Baseline (Phase 1)                  AUC: {baseline_auc:.4f}  (+ 0.00%)")
        print(f"  Phase 2 - Partial (4 tables)        AUC: {phase2_partial_auc:.4f}  (+ {((phase2_partial_auc/baseline_auc - 1) * 100):.2f}%)")
        print(f"  Phase 2 - Complete (7 tables)       AUC: {phase2_complete_auc:.4f}  (+ {((phase2_complete_auc/baseline_auc - 1) * 100):.2f}%)")
        print(f"  Phase 3 - Optimized + Interactions  AUC: {ensemble_auc:.4f}  (+ {((ensemble_auc/baseline_auc - 1) * 100):.2f}%)")
        
        improvement_from_phase2 = ensemble_auc - phase2_complete_auc
        print(f"\n🔥 Improvement from Phase 2 Complete:")
        print(f"   +{improvement_from_phase2:.4f} AUC ({(improvement_from_phase2/phase2_complete_auc * 100):+.2f}%)")
        
        # Success evaluation
        target_auc = 0.80
        if ensemble_auc >= target_auc:
            print(f"\n✅ SUCCESS! Achieved target of {target_auc:.2f}+ AUC!")
            print("🎉 Ready to move to Phase 4: Deployment!")
        else:
            gap = target_auc - ensemble_auc
            print(f"\n⚠️  Close! Gap to target: -{gap:.4f} AUC ({(gap/target_auc * 100):.2f}%)")
            print("\n💡 Additional improvements to try:")
            print("   1. More interaction features (current: 20)")
            print("   2. Feature engineering on underutilized tables")
            print("   3. Deep learning model (TabNet, FT-Transformer)")
            print("   4. More sophisticated stacking (Level-2 features)")
        
        # ===== SAVE MODELS =====
        print("\n" + "=" * 70)
        print("💾 Saving Models and Artifacts")
        print("=" * 70)
        
        models_dir = Path(config['models']['save_path'])
        models_dir.mkdir(exist_ok=True, parents=True)
        
        # Save LightGBM
        lgb_model.booster_.save_model(str(models_dir / 'lightgbm_phase3_optimized.txt'))
        logger.info("✓ Saved LightGBM model")
        
        # Save XGBoost
        xgb_model.save_model(str(models_dir / 'xgboost_phase3_optimized.json'))
        logger.info("✓ Saved XGBoost model")
        
        # Save CatBoost if available
        if catboost_available and cb_val_pred is not None:
            cb_model.save_model(str(models_dir / 'catboost_phase3_optimized.cbm'))
            logger.info("✓ Saved CatBoost model")
        
        # Save feature names
        with open(models_dir / 'feature_names_phase3.txt', 'w') as f:
            f.write('\n'.join(all_feature_names))
        logger.info("✓ Saved feature names")
        
        # Save feature importance
        feature_importance = pd.DataFrame({
            'feature': all_feature_names,
            'importance': lgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv(models_dir / 'feature_importance_phase3.csv', index=False)
        logger.info("✓ Saved feature importance")
        
        print("\n" + "=" * 70)
        print("✅ Phase 3 Complete!")
        print("=" * 70)
        
        return {
            'lgb_auc': lgb_val_auc,
            'xgb_auc': xgb_val_auc,
            'cb_auc': cb_val_auc,
            'ensemble_auc': ensemble_auc,
            'improvement': improvement_from_phase2,
            'target_achieved': ensemble_auc >= target_auc
        }


if __name__ == '__main__':
    results = main()
