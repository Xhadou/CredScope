"""CredScope Business Metrics Module

Provides business metrics, ROI calculation, and financial analysis.
"""

from .metrics import (
    ROICalculator,
    BusinessMetricsCalculator,
    LoanParameters,
    ModelPerformance
)

__all__ = [
    'ROICalculator',
    'BusinessMetricsCalculator',
    'LoanParameters',
    'ModelPerformance',
]
