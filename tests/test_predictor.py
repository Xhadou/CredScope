"""Tests for Credit Risk Predictor

Run with: pytest tests/test_predictor.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.credscope.api.predictor import CreditRiskPredictor


class TestPredictorInitialization:
    """Test predictor initialization"""

    def test_predictor_creation(self):
        """Test predictor can be created"""
        predictor = CreditRiskPredictor(models_dir="models")
        assert predictor is not None
        assert predictor.loaded is False

    def test_predictor_models_dir(self):
        """Test predictor uses correct models directory"""
        predictor = CreditRiskPredictor(models_dir="custom_models")
        assert str(predictor.models_dir).endswith("custom_models")


class TestInteractionFeatures:
    """Test interaction feature creation"""

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing"""
        return pd.DataFrame({
            'EXT_SOURCE_1': [0.5, 0.6, 0.7],
            'EXT_SOURCE_2': [0.6, 0.7, 0.8],
            'EXT_SOURCE_3': [0.4, 0.5, 0.6],
            'AMT_INCOME_TOTAL': [100000, 150000, 200000],
            'AMT_CREDIT': [300000, 400000, 500000],
            'DAYS_BIRTH': [-10000, -15000, -20000],
            'DAYS_EMPLOYED': [-2000, -3000, -4000],
            'CODE_GENDER': [0, 1, 0]
        })

    def test_create_interaction_features(self, sample_data):
        """Test interaction feature creation"""
        predictor = CreditRiskPredictor()
        result = predictor.create_interaction_features(sample_data)

        # Check new features were added
        assert len(result.columns) > len(sample_data.columns)

        # Check specific interaction features exist
        assert 'INT_EXT_AVG' in result.columns
        assert 'INT_EXT_23_MULT' in result.columns
        assert 'INT_AGE_CREDIT' in result.columns

    def test_interaction_no_nulls(self, sample_data):
        """Test interaction features don't introduce nulls"""
        predictor = CreditRiskPredictor()
        result = predictor.create_interaction_features(sample_data)

        # Check no infinite values
        assert not np.isinf(result.values).any()

        # Check NaN handling
        null_counts = result.isnull().sum()
        assert null_counts.max() == 0  # All NaNs should be filled

    def test_interaction_with_missing_columns(self):
        """Test interaction creation with missing columns"""
        predictor = CreditRiskPredictor()

        # Data missing some columns
        sparse_data = pd.DataFrame({
            'EXT_SOURCE_1': [0.5, 0.6],
            'AMT_INCOME_TOTAL': [100000, 150000]
        })

        result = predictor.create_interaction_features(sparse_data)

        # Should not crash, just skip missing interactions
        assert len(result) == len(sparse_data)


class TestPrediction:
    """Test prediction functionality"""

    @pytest.fixture
    def sample_applicant(self):
        """Sample applicant data"""
        return {
            'AMT_INCOME_TOTAL': 180000,
            'AMT_CREDIT': 500000,
            'AMT_ANNUITY': 25000,
            'DAYS_BIRTH': -15000,
            'DAYS_EMPLOYED': -3000,
            'CODE_GENDER': 1,
            'EXT_SOURCE_1': 0.65,
            'EXT_SOURCE_2': 0.72,
            'EXT_SOURCE_3': 0.58
        }

    def test_predict_single_format(self, sample_applicant):
        """Test single prediction returns correct format"""
        predictor = CreditRiskPredictor(models_dir="models")

        # Skip if models not available
        if not (Path("models") / "lightgbm_phase3_optimized.txt").exists():
            pytest.skip("Models not available")

        predictor.load_models()
        result = predictor.predict_single(sample_applicant)

        # Check all required fields
        assert 'default_probability' in result
        assert 'predicted_class' in result
        assert 'decision' in result
        assert 'confidence' in result
        assert 'risk_level' in result

        # Validate types
        assert isinstance(result['default_probability'], float)
        assert isinstance(result['predicted_class'], int)
        assert isinstance(result['decision'], str)
        assert isinstance(result['confidence'], float)
        assert isinstance(result['risk_level'], str)

        # Validate ranges
        assert 0 <= result['default_probability'] <= 1
        assert result['predicted_class'] in [0, 1]
        assert result['decision'] in ['APPROVE', 'REJECT']
        assert 0 <= result['confidence'] <= 1

    def test_risk_level_categorization(self):
        """Test risk level categorization"""
        predictor = CreditRiskPredictor()

        assert predictor._get_risk_level(0.1) == "VERY_LOW"
        assert predictor._get_risk_level(0.3) == "LOW"
        assert predictor._get_risk_level(0.5) == "MEDIUM"
        assert predictor._get_risk_level(0.7) == "HIGH"
        assert predictor._get_risk_level(0.9) == "VERY_HIGH"


class TestModelLoading:
    """Test model loading functionality"""

    def test_load_models_with_existing_files(self):
        """Test loading models when files exist"""
        if not (Path("models") / "lightgbm_phase3_optimized.txt").exists():
            pytest.skip("Models not available")

        predictor = CreditRiskPredictor(models_dir="models")
        predictor.load_models()

        assert predictor.loaded is True
        assert predictor.lgb_model is not None
        assert predictor.xgb_model is not None

    def test_load_models_missing_directory(self):
        """Test loading models from non-existent directory"""
        predictor = CreditRiskPredictor(models_dir="nonexistent_dir")

        with pytest.raises(FileNotFoundError):
            predictor.load_models()

    def test_prediction_without_loading(self):
        """Test prediction fails if models not loaded"""
        predictor = CreditRiskPredictor()

        data = pd.DataFrame({'AMT_INCOME_TOTAL': [100000]})

        with pytest.raises(RuntimeError):
            predictor.predict_proba(data.values)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
