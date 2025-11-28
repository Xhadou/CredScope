"""Pytest configuration and fixtures

Shared fixtures for all tests.
"""

import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def models_dir():
    """Path to models directory"""
    return Path("models")


@pytest.fixture(scope="session")
def models_available(models_dir):
    """Check if models are available"""
    lgb_path = models_dir / "lightgbm_model.txt"
    xgb_path = models_dir / "xgboost_model.json"
    return lgb_path.exists() and xgb_path.exists()
