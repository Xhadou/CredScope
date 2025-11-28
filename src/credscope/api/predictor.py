"""Model Serving Wrapper for CredScope

This module provides a unified interface for loading models and making predictions.
"""

import joblib
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

logger = logging.getLogger(__name__)


class CreditRiskPredictor:
    """Unified predictor for credit risk assessment using ensemble models"""

    def __init__(self, models_dir: str = "models"):
        """Initialize predictor

        Args:
            models_dir: Directory containing trained model files
        """
        self.models_dir = Path(models_dir)
        self.lgb_model = None
        self.xgb_model = None
        self.catb_model = None
        self.ensemble_weights = None  # Dict of model weights for ensemble
        self.feature_names = None
        self.loaded = False

    def load_models(self):
        """Load all trained models"""
        logger.info("Loading trained models...")

        try:
            # Load LightGBM (prefer phase3_optimized version)
            lgb_path = self.models_dir / "lightgbm_phase3_optimized.txt"
            if not lgb_path.exists():
                lgb_path = self.models_dir / "lightgbm_model.txt"
            if lgb_path.exists():
                self.lgb_model = lgb.Booster(model_file=str(lgb_path))
                logger.info(f"✓ LightGBM model loaded from {lgb_path.name}")
            else:
                raise FileNotFoundError(f"LightGBM model not found")

            # Load XGBoost (prefer phase3_optimized version)
            xgb_path = self.models_dir / "xgboost_phase3_optimized.json"
            if not xgb_path.exists():
                xgb_path = self.models_dir / "xgboost_model.json"
            if xgb_path.exists():
                self.xgb_model = xgb.Booster()
                self.xgb_model.load_model(str(xgb_path))
                logger.info(f"✓ XGBoost model loaded from {xgb_path.name}")
            else:
                raise FileNotFoundError(f"XGBoost model not found")

            # Load CatBoost (prefer phase3_optimized version)
            catb_path = self.models_dir / "catboost_phase3_optimized.cbm"
            if not catb_path.exists():
                catb_path = self.models_dir / "catboost_model.cbm"
            if catb_path.exists() and HAS_CATBOOST:
                self.catb_model = CatBoostClassifier()
                self.catb_model.load_model(str(catb_path))
                logger.info(f"✓ CatBoost model loaded from {catb_path.name}")
            else:
                logger.warning("CatBoost model not available - using 2-model ensemble")

            # Load ensemble weights dict
            meta_path = self.models_dir / "ensemble_meta_model.pkl"
            if meta_path.exists():
                loaded = joblib.load(meta_path)
                if isinstance(loaded, dict):
                    self.ensemble_weights = loaded
                    logger.info(f"✓ Ensemble weights loaded: {self.ensemble_weights}")
                else:
                    logger.warning("Unexpected meta-model format - using equal weights")
                    self.ensemble_weights = None
            else:
                logger.warning("Ensemble weights not found - using equal weights")
                self.ensemble_weights = None

            # Load feature names from XGBoost model (canonical order)
            # XGBoost stores feature names in the order they were trained
            if self.xgb_model and self.xgb_model.feature_names:
                self.feature_names = self.xgb_model.feature_names
                logger.info(f"✓ Feature names loaded from XGBoost model ({len(self.feature_names)} features)")
            else:
                # Fallback to feature importance file
                importance_path = self.models_dir / "feature_importance.csv"
                if importance_path.exists():
                    importance_df = pd.read_csv(importance_path)
                    self.feature_names = sorted(importance_df['feature'].tolist())
                    logger.info(f"✓ Feature names loaded from CSV ({len(self.feature_names)} features)")

            self.loaded = True
            logger.info("✅ All models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features for prediction

        Args:
            df: DataFrame with base features

        Returns:
            DataFrame with interaction features added
        """
        interactions = pd.DataFrame(index=df.index)

        # 1. External source interactions
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

        # 4. Payment behavior interactions
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

        # 7. Gender-based interactions
        if 'CODE_GENDER' in df.columns:
            interactions['INT_GENDER_INCOME'] = df['CODE_GENDER'] * df.get('AMT_INCOME_TOTAL', 0)
            interactions['INT_GENDER_CREDIT'] = df['CODE_GENDER'] * df.get('AMT_CREDIT', 0)

        # Handle infinities and NaNs
        interactions = interactions.replace([np.inf, -np.inf], np.nan)

        # Fill NaN with median (or 0 if median is NaN)
        for col in interactions.columns:
            median_val = interactions[col].median()
            if pd.isna(median_val):
                median_val = 0
            interactions[col] = interactions[col].fillna(median_val)

        # Combine with original dataframe
        result = pd.concat([df, interactions], axis=1)

        return result

    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for prediction

        Args:
            data: Raw input data

        Returns:
            Prepared feature array
        """
        # Create interaction features
        data_with_interactions = self.create_interaction_features(data)

        # Ensure all required features are present
        if self.feature_names:
            missing_features = set(self.feature_names) - set(data_with_interactions.columns)
            if missing_features:
                logger.warning(f"Missing {len(missing_features)} features - filling with 0")
                # Build all missing features at once using pd.concat for efficiency
                missing_df = pd.DataFrame(
                    {feat: [0.0] * len(data_with_interactions) for feat in missing_features},
                    index=data_with_interactions.index
                )
                data_with_interactions = pd.concat([data_with_interactions, missing_df], axis=1)

            # Select features in correct order and return DataFrame (not values)
            # This preserves feature names for XGBoost
            X = data_with_interactions[self.feature_names].copy()
        else:
            # If feature names not available, use all columns
            X = data_with_interactions.copy()

        # Ensure all columns are numeric (float64) to avoid XGBoost dtype errors
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)
        
        return X

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities from ensemble

        Args:
            X: Prepared feature DataFrame

        Returns:
            Array of default probabilities
        """
        if not self.loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Get predictions from each model
        # LightGBM accepts DataFrame or array
        pred_lgb = self.lgb_model.predict(X.values)
        
        # XGBoost - pass DataFrame with feature names matching model's expected order
        dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))
        pred_xgb = self.xgb_model.predict(dmatrix)

        if self.catb_model:
            pred_catb = self.catb_model.predict_proba(X.values)[:, 1]

            # Combine predictions using weighted average
            if self.ensemble_weights:
                w_lgb = self.ensemble_weights.get('lightgbm', 1/3)
                w_xgb = self.ensemble_weights.get('xgboost', 1/3)
                w_catb = self.ensemble_weights.get('catboost', 1/3)
                total_w = w_lgb + w_xgb + w_catb
                final_pred = (w_lgb * pred_lgb + w_xgb * pred_xgb + w_catb * pred_catb) / total_w
            else:
                # Equal weights
                final_pred = (pred_lgb + pred_xgb + pred_catb) / 3
        else:
            # 2-model ensemble
            if self.ensemble_weights:
                w_lgb = self.ensemble_weights.get('lightgbm', 0.5)
                w_xgb = self.ensemble_weights.get('xgboost', 0.5)
                total_w = w_lgb + w_xgb
                final_pred = (w_lgb * pred_lgb + w_xgb * pred_xgb) / total_w
            else:
                final_pred = (pred_lgb + pred_xgb) / 2

        return final_pred

    def predict(
        self,
        data: pd.DataFrame,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions for given data

        Args:
            data: Input DataFrame with applicant features
            threshold: Classification threshold (default 0.5)

        Returns:
            Tuple of (binary predictions, probabilities)
        """
        if not self.loaded:
            self.load_models()

        # Prepare features
        X = self.prepare_features(data)

        # Get probabilities
        proba = self.predict_proba(X)

        # Apply threshold
        predictions = (proba >= threshold).astype(int)

        return predictions, proba

    def _preprocess_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess input data to convert strings to numeric values
        
        Args:
            data: Raw input dictionary
            
        Returns:
            Preprocessed dictionary with numeric values
        """
        processed = data.copy()
        
        # Remove None/null values - they'll be filled with 0 later
        keys_to_remove = [k for k, v in processed.items() if v is None]
        for k in keys_to_remove:
            del processed[k]
        
        # Convert Y/N flags to 1/0
        flag_mappings = {
            'FLAG_OWN_CAR': {'Y': 1, 'N': 0, 'y': 1, 'n': 0},
            'FLAG_OWN_REALTY': {'Y': 1, 'N': 0, 'y': 1, 'n': 0},
        }
        
        for field, mapping in flag_mappings.items():
            if field in processed and isinstance(processed[field], str):
                processed[field] = mapping.get(processed[field], 0)
        
        # Convert categorical fields to numeric (one-hot style or ordinal)
        # Gender: Female=0, Male=1
        if 'CODE_GENDER' in processed and isinstance(processed['CODE_GENDER'], str):
            gender_map = {'F': 0, 'M': 1, 'Female': 0, 'Male': 1}
            processed['CODE_GENDER'] = gender_map.get(processed['CODE_GENDER'], 0)
        
        # Remove string categorical fields that the model doesn't use directly
        # (the feature engineering creates derived features from these)
        categorical_to_remove = [
            'NAME_CONTRACT_TYPE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
            'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE',
            'ORGANIZATION_TYPE'
        ]
        for field in categorical_to_remove:
            if field in processed and isinstance(processed[field], str):
                del processed[field]
        
        return processed

    def predict_single(
        self,
        applicant_data: Dict[str, Any],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Make prediction for single applicant

        Args:
            applicant_data: Dictionary with applicant features
            threshold: Classification threshold

        Returns:
            Dictionary with prediction results
        """
        # Preprocess input data
        processed_data = self._preprocess_input(applicant_data)
        
        # Convert to DataFrame
        df = pd.DataFrame([processed_data])

        # Make prediction
        predictions, proba = self.predict(df, threshold)

        # Format result with new thresholds:
        # < 20%: APPROVE, 20-50%: REVIEW, > 50%: REJECT
        prob = float(proba[0])
        if prob < 0.20:
            decision = 'APPROVE'
        elif prob < 0.50:
            decision = 'REVIEW'
        else:
            decision = 'REJECT'
        
        result = {
            'default_probability': prob,
            'predicted_class': int(predictions[0]),
            'decision': decision,
            'confidence': self._calculate_confidence(prob),
            'risk_level': self._get_risk_level(prob)
        }

        return result

    def _calculate_confidence(self, probability: float) -> float:
        """Calculate confidence based on distance from decision boundaries"""
        # Boundaries at 0.20 and 0.50
        if probability < 0.20:
            # Distance from 0.20 boundary
            return float(1.0 - (probability / 0.20))
        elif probability < 0.50:
            # In review zone - confidence is lower
            dist_from_center = abs(probability - 0.35) / 0.15
            return float(0.3 + 0.4 * dist_from_center)
        else:
            # Distance from 0.50 boundary  
            return float(min(1.0, (probability - 0.50) / 0.50))

    def _get_risk_level(self, probability: float) -> str:
        """Categorize risk level based on probability

        Args:
            probability: Default probability

        Returns:
            Risk level string
        """
        if probability < 0.10:
            return "VERY_LOW"
        elif probability < 0.20:
            return "LOW"
        elif probability < 0.35:
            return "MEDIUM"
        elif probability < 0.50:
            return "HIGH"
        else:
            return "VERY_HIGH"

    def explain_prediction(
        self,
        applicant_data: Dict[str, Any],
        top_n: int = 10
    ) -> Dict[str, Any]:
        """Get explanation for a prediction using SHAP

        Args:
            applicant_data: Dictionary with applicant features
            top_n: Number of top features to return

        Returns:
            Dictionary with prediction and top contributing features
        """
        # Preprocess input data
        processed_data = self._preprocess_input(applicant_data)
        
        # Make prediction (using already preprocessed data)
        result = self.predict_single(applicant_data)

        # Prepare features using preprocessed data
        df = pd.DataFrame([processed_data])
        X = self.prepare_features(df)

        try:
            import shap
            
            # Use TreeExplainer with model_output="raw" to avoid objective lookup issues
            explainer = shap.TreeExplainer(
                self.lgb_model,
                feature_perturbation="tree_path_dependent"
            )
            shap_values = explainer.shap_values(X.values)

            # For binary classification
            if isinstance(shap_values, list):
                shap_values = shap_values[1][0]
            else:
                shap_values = shap_values[0]

            # Get top features by absolute SHAP value
            abs_shap = np.abs(shap_values)
            top_indices = np.argsort(abs_shap)[-top_n:][::-1]
            
            # Calculate total absolute SHAP for percentage calculation
            total_abs_shap = np.sum(abs_shap)
            if total_abs_shap == 0:
                total_abs_shap = 1.0
            
            feature_importance = []
            for idx in top_indices:
                if self.feature_names:
                    feat_name = self.feature_names[idx]
                else:
                    feat_name = f"feature_{idx}"
                
                # Calculate percentage contribution (sums to ~100% for top features shown)
                percentage = (abs_shap[idx] / total_abs_shap) * 100

                feature_importance.append({
                    'feature': feat_name,
                    'shap_value': float(shap_values[idx]),
                    'importance': float(percentage),
                    'feature_value': float(X.values[0, idx]),
                    'impact': 'increases' if shap_values[idx] > 0 else 'decreases'
                })

            result['top_features'] = feature_importance
            
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            # Fallback: use feature importance from model instead
            result['top_features'] = self._get_fallback_feature_importance(X, top_n)

        return result

    def _get_fallback_feature_importance(self, X: pd.DataFrame, top_n: int = 10) -> list:
        """Get feature importance as fallback when SHAP fails"""
        try:
            # Use LightGBM's built-in feature importance
            importance = self.lgb_model.feature_importance(importance_type='gain')
            
            # Get top features
            top_indices = np.argsort(importance)[-top_n:][::-1]
            
            # Calculate total for percentage
            total_importance = np.sum(importance[top_indices])
            if total_importance == 0:
                total_importance = 1.0
            
            feature_importance = []
            for idx in top_indices:
                feat_name = self.feature_names[idx] if self.feature_names else f"feature_{idx}"
                percentage = (importance[idx] / total_importance) * 100
                feature_importance.append({
                    'feature': feat_name,
                    'shap_value': float(importance[idx]),
                    'importance': float(percentage),
                    'feature_value': float(X.values[0, idx]),
                    'impact': 'important'
                })
            return feature_importance
        except Exception:
            return []


# Global predictor instance (for API usage)
_predictor = None


def get_predictor(models_dir: str = "models") -> CreditRiskPredictor:
    """Get or create global predictor instance

    Args:
        models_dir: Directory containing models

    Returns:
        CreditRiskPredictor instance
    """
    global _predictor
    if _predictor is None:
        _predictor = CreditRiskPredictor(models_dir)
        _predictor.load_models()
    return _predictor


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("CredScope Model Predictor")
    print("=" * 50)
    print("\nUsage:")
    print("""
    from src.credscope.api.predictor import CreditRiskPredictor

    # Initialize predictor
    predictor = CreditRiskPredictor(models_dir='models')
    predictor.load_models()

    # Make prediction
    applicant = {
        'AMT_INCOME_TOTAL': 180000,
        'AMT_CREDIT': 500000,
        'DAYS_BIRTH': -15000,
        # ... more features
    }

    result = predictor.predict_single(applicant)
    print(result)
    # {'default_probability': 0.23, 'decision': 'APPROVE', ...}
    """)
