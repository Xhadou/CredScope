"""Model Drift Detection and Performance Monitoring

Detects data drift, concept drift, and model performance degradation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats
from collections import deque
import logging
import json

logger = logging.getLogger(__name__)


class DataDriftDetector:
    """Detect data drift using statistical tests"""

    def __init__(self, reference_data: pd.DataFrame, significance_level: float = 0.05):
        """Initialize drift detector

        Args:
            reference_data: Reference dataset (training data)
            significance_level: Significance level for statistical tests
        """
        self.reference_data = reference_data
        self.significance_level = significance_level
        self.reference_stats = self._compute_statistics(reference_data)

    def _compute_statistics(self, data: pd.DataFrame) -> Dict:
        """Compute statistical properties of data

        Args:
            data: Input dataframe

        Returns:
            Dictionary of statistics
        """
        stats_dict = {}

        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                stats_dict[col] = {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'median': data[col].median(),
                    'q25': data[col].quantile(0.25),
                    'q75': data[col].quantile(0.75)
                }

        return stats_dict

    def detect_drift_ks_test(
        self,
        current_data: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """Detect drift using Kolmogorov-Smirnov test

        Args:
            current_data: Current production data
            features: List of features to check (all if None)

        Returns:
            Dictionary with drift results per feature
        """
        if features is None:
            features = self.reference_data.columns.tolist()

        drift_results = {}

        for feature in features:
            if feature not in self.reference_data.columns:
                continue

            if not pd.api.types.is_numeric_dtype(self.reference_data[feature]):
                continue

            # Perform KS test
            statistic, p_value = stats.ks_2samp(
                self.reference_data[feature].dropna(),
                current_data[feature].dropna()
            )

            drift_detected = p_value < self.significance_level

            drift_results[feature] = {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'drift_detected': drift_detected,
                'drift_score': float(statistic),  # Higher = more drift
                'test': 'ks_test'
            }

        return drift_results

    def detect_drift_psi(
        self,
        current_data: pd.DataFrame,
        features: Optional[List[str]] = None,
        bins: int = 10
    ) -> Dict[str, Dict]:
        """Detect drift using Population Stability Index (PSI)

        Args:
            current_data: Current production data
            features: List of features to check
            bins: Number of bins for discretization

        Returns:
            Dictionary with PSI results per feature
        """
        if features is None:
            features = self.reference_data.columns.tolist()

        drift_results = {}

        for feature in features:
            if feature not in self.reference_data.columns:
                continue

            if not pd.api.types.is_numeric_dtype(self.reference_data[feature]):
                continue

            # Calculate PSI
            psi_value = self._calculate_psi(
                self.reference_data[feature].dropna(),
                current_data[feature].dropna(),
                bins
            )

            # PSI thresholds: <0.1 (no drift), 0.1-0.2 (moderate), >0.2 (significant)
            if psi_value < 0.1:
                drift_level = 'none'
            elif psi_value < 0.2:
                drift_level = 'moderate'
            else:
                drift_level = 'significant'

            drift_results[feature] = {
                'psi_value': float(psi_value),
                'drift_level': drift_level,
                'drift_detected': psi_value >= 0.2,
                'test': 'psi'
            }

        return drift_results

    def _calculate_psi(
        self,
        reference: pd.Series,
        current: pd.Series,
        bins: int
    ) -> float:
        """Calculate Population Stability Index

        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins

        Returns:
            PSI value
        """
        # Create bins based on reference data
        breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)

        # Calculate distributions
        ref_counts, _ = np.histogram(reference, bins=breakpoints)
        curr_counts, _ = np.histogram(current, bins=breakpoints)

        # Avoid division by zero
        ref_counts = ref_counts + 0.0001
        curr_counts = curr_counts + 0.0001

        # Normalize
        ref_dist = ref_counts / ref_counts.sum()
        curr_dist = curr_counts / curr_counts.sum()

        # Calculate PSI
        psi = np.sum((curr_dist - ref_dist) * np.log(curr_dist / ref_dist))

        return psi


class PerformanceMonitor:
    """Monitor model performance over time"""

    def __init__(self, window_size: int = 1000):
        """Initialize performance monitor

        Args:
            window_size: Number of recent predictions to track
        """
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.actuals = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.probabilities = deque(maxlen=window_size)

    def add_prediction(
        self,
        prediction: int,
        probability: float,
        actual: Optional[int] = None,
        timestamp: Optional[datetime] = None
    ):
        """Add prediction to monitor

        Args:
            prediction: Predicted class (0 or 1)
            probability: Prediction probability
            actual: Actual label (if available)
            timestamp: Prediction timestamp
        """
        self.predictions.append(prediction)
        self.probabilities.append(probability)
        self.actuals.append(actual)
        self.timestamps.append(timestamp or datetime.utcnow())

    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics

        Returns:
            Dictionary with metrics
        """
        # Filter out predictions without actuals
        paired_data = [
            (pred, actual, prob)
            for pred, actual, prob in zip(self.predictions, self.actuals, self.probabilities)
            if actual is not None
        ]

        if not paired_data:
            return {
                'error': 'No actual labels available',
                'sample_size': 0
            }

        predictions, actuals, probabilities = zip(*paired_data)

        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

        metrics = {
            'sample_size': len(predictions),
            'accuracy': float(accuracy_score(actuals, predictions)),
            'precision': float(precision_score(actuals, predictions, zero_division=0)),
            'recall': float(recall_score(actuals, predictions, zero_division=0)),
            'auc': float(roc_auc_score(actuals, probabilities)),
            'approval_rate': float(np.mean([1 - p for p in predictions])),  # APPROVE = 0
            'avg_probability': float(np.mean(probabilities))
        }

        return metrics

    def detect_performance_drift(
        self,
        baseline_auc: float = 0.7908,
        threshold: float = 0.05
    ) -> Dict:
        """Detect if performance has degraded

        Args:
            baseline_auc: Expected baseline AUC
            threshold: Acceptable degradation threshold

        Returns:
            Dictionary with drift detection results
        """
        current_metrics = self.calculate_metrics()

        if 'error' in current_metrics:
            return current_metrics

        current_auc = current_metrics['auc']
        degradation = baseline_auc - current_auc

        drift_detected = degradation > threshold

        return {
            'baseline_auc': baseline_auc,
            'current_auc': current_auc,
            'degradation': float(degradation),
            'degradation_percent': float(degradation / baseline_auc * 100),
            'drift_detected': drift_detected,
            'threshold': threshold,
            'status': 'degraded' if drift_detected else 'stable',
            'sample_size': current_metrics['sample_size']
        }

    def get_time_series_metrics(
        self,
        window_days: int = 7,
        bucket_hours: int = 24
    ) -> List[Dict]:
        """Get metrics over time

        Args:
            window_days: Number of days to analyze
            bucket_hours: Hours per bucket

        Returns:
            List of time-bucketed metrics
        """
        if not self.timestamps:
            return []

        # Create time buckets
        now = datetime.utcnow()
        cutoff = now - timedelta(days=window_days)

        # Filter recent data
        recent_data = [
            (ts, pred, actual, prob)
            for ts, pred, actual, prob in zip(
                self.timestamps, self.predictions, self.actuals, self.probabilities
            )
            if ts >= cutoff and actual is not None
        ]

        if not recent_data:
            return []

        # Group by time buckets
        bucket_size = timedelta(hours=bucket_hours)
        buckets = {}

        for ts, pred, actual, prob in recent_data:
            bucket_start = ts.replace(minute=0, second=0, microsecond=0)
            bucket_start = bucket_start - timedelta(hours=bucket_start.hour % bucket_hours)

            if bucket_start not in buckets:
                buckets[bucket_start] = {
                    'predictions': [],
                    'actuals': [],
                    'probabilities': []
                }

            buckets[bucket_start]['predictions'].append(pred)
            buckets[bucket_start]['actuals'].append(actual)
            buckets[bucket_start]['probabilities'].append(prob)

        # Calculate metrics per bucket
        time_series = []
        for bucket_start in sorted(buckets.keys()):
            data = buckets[bucket_start]

            from sklearn.metrics import accuracy_score, roc_auc_score

            metrics = {
                'timestamp': bucket_start.isoformat(),
                'sample_size': len(data['predictions']),
                'accuracy': float(accuracy_score(data['actuals'], data['predictions'])),
                'auc': float(roc_auc_score(data['actuals'], data['probabilities'])),
                'approval_rate': float(np.mean([1 - p for p in data['predictions']]))
            }

            time_series.append(metrics)

        return time_series


class AlertManager:
    """Manage drift and performance alerts"""

    def __init__(self):
        self.alerts = []

    def check_drift_alerts(
        self,
        drift_results: Dict[str, Dict],
        threshold: float = 0.2
    ) -> List[Dict]:
        """Check for drift alerts

        Args:
            drift_results: Drift detection results
            threshold: Alert threshold

        Returns:
            List of alerts
        """
        alerts = []

        for feature, result in drift_results.items():
            if result.get('drift_detected'):
                severity = 'critical' if result.get('drift_score', 0) > 0.5 else 'warning'

                alert = {
                    'type': 'data_drift',
                    'severity': severity,
                    'feature': feature,
                    'message': f"Data drift detected in feature '{feature}'",
                    'details': result,
                    'timestamp': datetime.utcnow().isoformat()
                }

                alerts.append(alert)

        return alerts

    def check_performance_alerts(
        self,
        performance_results: Dict,
        critical_threshold: float = 0.1
    ) -> List[Dict]:
        """Check for performance degradation alerts

        Args:
            performance_results: Performance monitoring results
            critical_threshold: Critical degradation threshold

        Returns:
            List of alerts
        """
        alerts = []

        if performance_results.get('drift_detected'):
            degradation = performance_results.get('degradation', 0)

            severity = 'critical' if degradation > critical_threshold else 'warning'

            alert = {
                'type': 'performance_degradation',
                'severity': severity,
                'message': f"Model performance degraded by {degradation:.1%}",
                'details': performance_results,
                'timestamp': datetime.utcnow().isoformat()
            }

            alerts.append(alert)

        return alerts


if __name__ == "__main__":
    # Demo usage
    print("Drift Detection Demo")
    print("=" * 50)

    # Generate sample data
    np.random.seed(42)

    reference_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.normal(50, 10, 1000),
        'feature3': np.random.exponential(2, 1000)
    })

    # Current data with drift in feature1
    current_data = pd.DataFrame({
        'feature1': np.random.normal(110, 15, 500),  # Mean shifted
        'feature2': np.random.normal(50, 10, 500),   # No drift
        'feature3': np.random.exponential(2, 500)
    })

    # Test drift detection
    detector = DataDriftDetector(reference_data)

    ks_results = detector.detect_drift_ks_test(current_data)
    print("\nKS Test Results:")
    for feature, result in ks_results.items():
        print(f"  {feature}: drift={result['drift_detected']}, p-value={result['p_value']:.4f}")

    psi_results = detector.detect_drift_psi(current_data)
    print("\nPSI Results:")
    for feature, result in psi_results.items():
        print(f"  {feature}: PSI={result['psi_value']:.4f}, level={result['drift_level']}")

    # Test performance monitoring
    monitor = PerformanceMonitor(window_size=100)

    # Add sample predictions with actuals
    for i in range(100):
        pred = np.random.binomial(1, 0.3)
        prob = np.random.beta(2, 5)
        actual = np.random.binomial(1, 0.25)
        monitor.add_prediction(pred, prob, actual)

    metrics = monitor.calculate_metrics()
    print(f"\nPerformance Metrics:")
    print(json.dumps(metrics, indent=2))

    drift_check = monitor.detect_performance_drift()
    print(f"\nPerformance Drift Check:")
    print(json.dumps(drift_check, indent=2))
