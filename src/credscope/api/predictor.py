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
from sklearn.linear_model import LogisticRegression

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
        self.meta_model = None
        self.feature_names = None
        self.loaded = False

    def load_models(self):
        """Load all trained models"""
        logger.info("Loading trained models...")

        try:
            # Load LightGBM
            lgb_path = self.models_dir / "lightgbm_phase3_optimized.txt"
            if lgb_path.exists():
                self.lgb_model = lgb.Booster(model_file=str(lgb_path))
                logger.info("✓ LightGBM model loaded")
            else:
                raise FileNotFoundError(f"LightGBM model not found at {lgb_path}")

            # Load XGBoost
            xgb_path = self.models_dir / "xgboost_phase3_optimized.json"
            if xgb_path.exists():
                self.xgb_model = xgb.Booster()
                self.xgb_model.load_model(str(xgb_path))
                logger.info("✓ XGBoost model loaded")
            else:
                raise FileNotFoundError(f"XGBoost model not found at {xgb_path}")

            # Load CatBoost if available
            catb_path = self.models_dir / "catboost_phase3_optimized.cbm"
            if catb_path.exists() and HAS_CATBOOST:
                self.catb_model = CatBoostClassifier()
                self.catb_model.load_model(str(catb_path))
                logger.info("✓ CatBoost model loaded")
            else:
                logger.warning("CatBoost model not available - using 2-model ensemble")

            # Load meta-model (ensemble weights)
            # If not available, use equal weights
            meta_path = self.models_dir / "ensemble_weights_phase3.pkl"
            if meta_path.exists():
                self.meta_model = joblib.load(meta_path)
                logger.info("✓ Ensemble meta-model loaded")
            else:
                logger.warning("Meta-model not found - using equal weights")
                self.meta_model = None

            # Load feature names from feature importance file
            importance_path = self.models_dir / "feature_importance_phase3.csv"
            if importance_path.exists():
                importance_df = pd.read_csv(importance_path)
                self.feature_names = importance_df['feature'].tolist()
                logger.info(f"✓ Feature names loaded ({len(self.feature_names)} features)")

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
                for feat in missing_features:
                    data_with_interactions[feat] = 0

            # Select features in correct order
            X = data_with_interactions[self.feature_names].values
        else:
            # If feature names not available, use all columns
            X = data_with_interactions.values

        return X

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities from ensemble

        Args:
            X: Prepared feature array

        Returns:
            Array of default probabilities
        """
        if not self.loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Get predictions from each model
        pred_lgb = self.lgb_model.predict(X)
        pred_xgb = self.xgb_model.predict(xgb.DMatrix(X))

        if self.catb_model:
            pred_catb = self.catb_model.predict_proba(X)[:, 1]

            # Combine predictions
            if self.meta_model:
                # Use meta-model
                meta_X = np.column_stack([pred_lgb, pred_xgb, pred_catb])
                final_pred = self.meta_model.predict_proba(meta_X)[:, 1]
            else:
                # Equal weights
                final_pred = (pred_lgb + pred_xgb + pred_catb) / 3
        else:
            # 2-model ensemble
            if self.meta_model:
                meta_X = np.column_stack([pred_lgb, pred_xgb])
                final_pred = self.meta_model.predict_proba(meta_X)[:, 1]
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
        # Convert to DataFrame
        df = pd.DataFrame([applicant_data])

        # Make prediction
        predictions, proba = self.predict(df, threshold)

        # Format result
        result = {
            'default_probability': float(proba[0]),
            'predicted_class': int(predictions[0]),
            'decision': 'REJECT' if predictions[0] == 1 else 'APPROVE',
            'confidence': float(abs(proba[0] - 0.5) * 2),  # 0-1 scale
            'risk_level': self._get_risk_level(proba[0])
        }

        return result

    def _get_risk_level(self, probability: float) -> str:
        """Categorize risk level based on probability

        Args:
            probability: Default probability

        Returns:
            Risk level string
        """
        if probability < 0.2:
            return "VERY_LOW"
        elif probability < 0.4:
            return "LOW"
        elif probability < 0.6:
            return "MEDIUM"
        elif probability < 0.8:
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
        import shap

        # Make prediction
        result = self.predict_single(applicant_data)

        # Prepare features
        df = pd.DataFrame([applicant_data])
        X = self.prepare_features(df)

        # Use LightGBM for SHAP (fastest)
        explainer = shap.TreeExplainer(self.lgb_model)
        shap_values = explainer.shap_values(X)

        # For binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1][0]
        else:
            shap_values = shap_values[0]

        # Get top features
        feature_importance = []
        for idx in np.argsort(np.abs(shap_values))[-top_n:][::-1]:
            if self.feature_names:
                feat_name = self.feature_names[idx]
            else:
                feat_name = f"feature_{idx}"

            feature_importance.append({
                'feature': feat_name,
                'shap_value': float(shap_values[idx]),
                'feature_value': float(X[0, idx]),
                'impact': 'increases' if shap_values[idx] > 0 else 'decreases'
            })

        result['top_features'] = feature_importance

        return result


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
