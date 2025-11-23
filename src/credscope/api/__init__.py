"""CredScope API Module

Provides REST API and prediction services for credit risk assessment.
"""

from .predictor import CreditRiskPredictor, get_predictor
from .main import app

__all__ = [
    'CreditRiskPredictor',
    'get_predictor',
    'app',
]
