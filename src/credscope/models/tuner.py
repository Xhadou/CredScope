"""Hyperparameter Tuning Utilities

This module provides Optuna-based hyperparameter optimization for LightGBM and XGBoost.
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


class LightGBMTuner:
    """Optuna-based hyperparameter tuner for LightGBM"""
    
    def __init__(self, X_train, y_train, X_val, y_val, n_trials: int = 200):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = None
        self.study = None
        
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization"""
        
        # Define hyperparameter search space
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': 42,
            
            # Tree structure
            'num_leaves': trial.suggest_int('num_leaves', 20, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 500),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 10.0, log=True),
            
            # Learning
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            
            # Sampling
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 0, 10),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            
            # Regularization
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            
            # Speed optimization
            'max_bin': trial.suggest_int('max_bin', 200, 300),
        }
        
        # Create datasets
        train_data = lgb.Dataset(self.X_train, label=self.y_train)
        val_data = lgb.Dataset(self.X_val, label=self.y_val, reference=train_data)
        
        # Train with early stopping
        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, 'auc')
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            valid_names=['valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                pruning_callback
            ]
        )
        
        # Predict and calculate AUC
        preds = model.predict(self.X_val)  # type: ignore[arg-type]
        score = roc_auc_score(self.y_val, preds)  # type: ignore[arg-type]
        
        return float(score)
    
    def optimize(self) -> Tuple[Dict, float]:
        """Run Optuna optimization
        
        Returns:
            Tuple of (best_params, best_score)
        """
        logger.info(f"Starting LightGBM optimization with {self.n_trials} trials...")
        
        # Create study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=20)
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name='lightgbm_optimization'
        )
        
        # Optimize
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
            n_jobs=1  # Use 1 to avoid conflicts
        )
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        logger.info(f"✓ Best LightGBM AUC: {self.best_score:.4f}")
        logger.info(f"✓ Best params: {self.best_params}")
        
        return self.best_params, self.best_score
    
    def get_best_model(self):
        """Train final model with best parameters"""
        if self.best_params is None:
            raise ValueError("Must run optimize() first")
        
        # Add fixed params
        params = {
            **self.best_params,
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': 42
        }
        
        # Train final model
        train_data = lgb.Dataset(self.X_train, label=self.y_train)
        val_data = lgb.Dataset(self.X_val, label=self.y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[lgb.early_stopping(stopping_rounds=100)]
        )
        
        return model


class XGBoostTuner:
    """Optuna-based hyperparameter tuner for XGBoost"""
    
    def __init__(self, X_train, y_train, X_val, y_val, n_trials: int = 200):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = None
        self.study = None
        
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization"""
        
        # Define hyperparameter search space
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'random_state': 42,
            
            # Tree structure
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_float('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 1e-8, 10.0, log=True),
            
            # Learning
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            
            # Sampling
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            
            # Regularization
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            
            # Speed
            'max_bin': trial.suggest_int('max_bin', 200, 300),
        }
        
        # Create DMatrix
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dval = xgb.DMatrix(self.X_val, label=self.y_val)
        
        # Train with pruning
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'valid-auc')
        
        model = xgb.train(
            params,
            dtrain,
            evals=[(dval, 'valid')],
            early_stopping_rounds=50,
            callbacks=[pruning_callback],
            verbose_eval=False
        )
        
        # Predict and calculate AUC
        preds = model.predict(dval)
        score = roc_auc_score(self.y_val, preds)
        
        return float(score)
    
    def optimize(self) -> Tuple[Dict, float]:
        """Run Optuna optimization
        
        Returns:
            Tuple of (best_params, best_score)
        """
        logger.info(f"Starting XGBoost optimization with {self.n_trials} trials...")
        
        # Create study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=20)
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name='xgboost_optimization'
        )
        
        # Optimize
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
            n_jobs=1
        )
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        logger.info(f"✓ Best XGBoost AUC: {self.best_score:.4f}")
        logger.info(f"✓ Best params: {self.best_params}")
        
        return self.best_params, self.best_score
    
    def get_best_model(self):
        """Train final model with best parameters"""
        if self.best_params is None:
            raise ValueError("Must run optimize() first")
        
        # Add fixed params
        params = {
            **self.best_params,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'random_state': 42
        }
        
        # Train final model
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dval = xgb.DMatrix(self.X_val, label=self.y_val)
        
        model = xgb.train(
            params,
            dtrain,
            evals=[(dtrain, 'train'), (dval, 'valid')],
            early_stopping_rounds=100,
            verbose_eval=False
        )
        
        return model


def optimize_ensemble_weights(predictions_dict: Dict[str, np.ndarray], 
                              y_true: np.ndarray,
                              n_trials: int = 100) -> Dict[str, float]:
    """Optimize ensemble weights using Optuna
    
    Args:
        predictions_dict: Dictionary of {model_name: predictions}
        y_true: True labels
        n_trials: Number of optimization trials
        
    Returns:
        Dictionary of optimized weights
    """
    logger.info("Optimizing ensemble weights...")
    
    model_names = list(predictions_dict.keys())
    predictions_array = np.column_stack([predictions_dict[name] for name in model_names])
    
    def objective(trial):
        # Suggest weights (they will be normalized)
        weights = [trial.suggest_float(f'weight_{name}', 0.0, 1.0) 
                  for name in model_names]
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Calculate weighted average
        ensemble_pred = predictions_array @ weights
        score = roc_auc_score(y_true, ensemble_pred)
        
        return float(score)
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    # Get best weights
    best_weights = {name: study.best_params[f'weight_{name}'] for name in model_names}
    
    # Normalize
    total = sum(best_weights.values())
    best_weights = {k: v/total for k, v in best_weights.items()}
    
    logger.info(f"✓ Optimized weights: {best_weights}")
    logger.info(f"✓ Best ensemble AUC: {study.best_value:.4f}")
    
    return best_weights


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Hyperparameter Tuning Utilities")
    print("=" * 50)
    print("\nUsage:")
    print("""
    from src.credscope.models.tuner import LightGBMTuner, XGBoostTuner
    
    # Optimize LightGBM
    lgb_tuner = LightGBMTuner(X_train, y_train, X_val, y_val, n_trials=200)
    best_params, best_score = lgb_tuner.optimize()
    best_model = lgb_tuner.get_best_model()
    
    # Optimize XGBoost
    xgb_tuner = XGBoostTuner(X_train, y_train, X_val, y_val, n_trials=200)
    best_params, best_score = xgb_tuner.optimize()
    best_model = xgb_tuner.get_best_model()
    """)