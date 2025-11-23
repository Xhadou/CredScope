"""Tests for CredScope API

Run with: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.credscope.api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_check(self):
        """Test health endpoint returns 200"""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "models_loaded" in data
        assert "version" in data


class TestRootEndpoint:
    """Test root endpoint"""

    def test_root(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "version" in data


class TestPredictionEndpoint:
    """Test prediction endpoints"""

    @pytest.fixture
    def sample_applicant(self):
        """Sample applicant data"""
        return {
            "AMT_INCOME_TOTAL": 180000,
            "AMT_CREDIT": 500000,
            "AMT_ANNUITY": 25000,
            "AMT_GOODS_PRICE": 450000,
            "DAYS_BIRTH": -15000,
            "DAYS_EMPLOYED": -3000,
            "CODE_GENDER": 1,
            "EXT_SOURCE_1": 0.65,
            "EXT_SOURCE_2": 0.72,
            "EXT_SOURCE_3": 0.58
        }

    def test_predict_endpoint(self, sample_applicant):
        """Test single prediction endpoint"""
        response = client.post("/predict", json=sample_applicant)

        # If models are not loaded, this might fail gracefully
        if response.status_code == 503:
            pytest.skip("Models not loaded - expected in test environment")

        assert response.status_code == 200

        data = response.json()
        assert "default_probability" in data
        assert "predicted_class" in data
        assert "decision" in data
        assert "risk_level" in data
        assert "confidence" in data
        assert "timestamp" in data

        # Validate types and ranges
        assert 0 <= data["default_probability"] <= 1
        assert data["predicted_class"] in [0, 1]
        assert data["decision"] in ["APPROVE", "REJECT"]
        assert 0 <= data["confidence"] <= 1

    def test_predict_with_explanation(self, sample_applicant):
        """Test prediction with explanation endpoint"""
        response = client.post("/predict/explain", json=sample_applicant)

        if response.status_code == 503:
            pytest.skip("Models not loaded - expected in test environment")

        assert response.status_code == 200

        data = response.json()
        assert "default_probability" in data
        assert "top_features" in data
        assert isinstance(data["top_features"], list)

        # Check feature format
        if len(data["top_features"]) > 0:
            feature = data["top_features"][0]
            assert "feature" in feature
            assert "shap_value" in feature
            assert "feature_value" in feature
            assert "impact" in feature

    def test_predict_missing_required_field(self):
        """Test prediction with missing required field"""
        incomplete_data = {
            "AMT_INCOME_TOTAL": 180000,
            # Missing AMT_CREDIT (required)
        }

        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422  # Validation error

    def test_predict_invalid_values(self):
        """Test prediction with invalid values"""
        invalid_data = {
            "AMT_INCOME_TOTAL": -100,  # Negative income (invalid)
            "AMT_CREDIT": 500000,
            "DAYS_BIRTH": -15000
        }

        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error

    def test_batch_prediction(self, sample_applicant):
        """Test batch prediction endpoint"""
        batch_request = {
            "applicants": [
                sample_applicant,
                {**sample_applicant, "AMT_INCOME_TOTAL": 200000},
                {**sample_applicant, "AMT_CREDIT": 600000}
            ]
        }

        response = client.post("/predict/batch", json=batch_request)

        if response.status_code == 503:
            pytest.skip("Models not loaded - expected in test environment")

        assert response.status_code == 200

        data = response.json()
        assert "predictions" in data
        assert "total_count" in data
        assert "approved_count" in data
        assert "rejected_count" in data

        assert data["total_count"] == 3
        assert len(data["predictions"]) == 3
        assert data["approved_count"] + data["rejected_count"] == 3


class TestModelInfoEndpoint:
    """Test model info endpoint"""

    def test_model_info(self):
        """Test model info endpoint"""
        response = client.get("/models/info")

        if response.status_code == 503:
            pytest.skip("Models not loaded - expected in test environment")

        assert response.status_code == 200

        data = response.json()
        assert "models_loaded" in data
        assert "lightgbm" in data
        assert "xgboost" in data
        assert "catboost" in data


class TestAPIDocumentation:
    """Test API documentation endpoints"""

    def test_openapi_schema(self):
        """Test OpenAPI schema is available"""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema

    def test_docs_page(self):
        """Test Swagger UI docs page"""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_page(self):
        """Test ReDoc docs page"""
        response = client.get("/redoc")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
