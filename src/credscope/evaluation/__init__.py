"""CredScope Model Evaluation Module

This module provides model evaluation and interpretation tools.
"""

from .explainer import SHAPExplainer, analyze_model_with_shap
from .fairness import FairnessAuditor, audit_model_fairness

__all__ = [
    'SHAPExplainer',
    'analyze_model_with_shap',
    'FairnessAuditor',
    'audit_model_fairness',
]