"""Prometheus Monitoring and Metrics

Provides metrics collection for API monitoring and observability.
"""

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, REGISTRY
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST
from fastapi import Response
import time
import psutil
import logging

logger = logging.getLogger(__name__)

# API Request Metrics
api_requests_total = Counter(
    'credscope_api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status_code']
)

api_request_duration_seconds = Histogram(
    'credscope_api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# Prediction Metrics
predictions_total = Counter(
    'credscope_predictions_total',
    'Total number of predictions made',
    ['decision', 'risk_level']
)

prediction_probability = Histogram(
    'credscope_prediction_probability',
    'Distribution of default probabilities',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

prediction_latency_seconds = Histogram(
    'credscope_prediction_latency_seconds',
    'Time taken to make a prediction',
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
)

# Model Metrics
model_loaded = Gauge(
    'credscope_model_loaded',
    'Whether models are loaded (1=loaded, 0=not loaded)',
    ['model_name']
)

model_prediction_errors = Counter(
    'credscope_model_prediction_errors_total',
    'Total number of model prediction errors',
    ['error_type']
)

# System Metrics
system_memory_usage_bytes = Gauge(
    'credscope_system_memory_usage_bytes',
    'System memory usage in bytes'
)

system_cpu_usage_percent = Gauge(
    'credscope_system_cpu_usage_percent',
    'System CPU usage percentage'
)

# API Key Metrics
active_api_keys = Gauge(
    'credscope_active_api_keys_total',
    'Number of active API keys',
    ['tier']
)

rate_limit_exceeded_total = Counter(
    'credscope_rate_limit_exceeded_total',
    'Total number of rate limit exceeded errors',
    ['tier']
)

# Application Info
app_info = Info(
    'credscope_app',
    'Application information'
)
app_info.info({
    'version': '1.0.0',
    'name': 'CredScope',
    'description': 'Alternative Credit Risk Assessment API'
})


class MetricsCollector:
    """Collect and manage application metrics"""

    def __init__(self):
        self.start_time = time.time()

    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record API request metrics

        Args:
            method: HTTP method
            endpoint: API endpoint
            status_code: Response status code
            duration: Request duration in seconds
        """
        api_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).inc()

        api_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)

    def record_prediction(
        self,
        decision: str,
        risk_level: str,
        probability: float,
        latency: float
    ):
        """Record prediction metrics

        Args:
            decision: APPROVE or REJECT
            risk_level: Risk level category
            probability: Default probability
            latency: Prediction latency in seconds
        """
        predictions_total.labels(
            decision=decision,
            risk_level=risk_level
        ).inc()

        prediction_probability.observe(probability)
        prediction_latency_seconds.observe(latency)

    def record_model_status(self, model_name: str, loaded: bool):
        """Record model load status

        Args:
            model_name: Name of the model
            loaded: Whether model is loaded
        """
        model_loaded.labels(model_name=model_name).set(1 if loaded else 0)

    def record_prediction_error(self, error_type: str):
        """Record prediction error

        Args:
            error_type: Type of error
        """
        model_prediction_errors.labels(error_type=error_type).inc()

    def record_rate_limit_exceeded(self, tier: str):
        """Record rate limit exceeded

        Args:
            tier: API key tier
        """
        rate_limit_exceeded_total.labels(tier=tier).inc()

    def update_system_metrics(self):
        """Update system resource metrics"""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            system_memory_usage_bytes.set(memory.used)

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            system_cpu_usage_percent.set(cpu_percent)

        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")

    def update_api_key_metrics(self, keys_by_tier: dict):
        """Update API key metrics

        Args:
            keys_by_tier: Dictionary of {tier: count}
        """
        for tier, count in keys_by_tier.items():
            active_api_keys.labels(tier=tier).set(count)

    def get_uptime_seconds(self) -> float:
        """Get application uptime in seconds"""
        return time.time() - self.start_time


# Global collector instance
metrics_collector = MetricsCollector()


def get_metrics() -> Response:
    """Get Prometheus metrics

    Returns:
        Response with metrics in Prometheus format
    """
    # Update system metrics before returning
    metrics_collector.update_system_metrics()

    # Generate metrics
    metrics = generate_latest(REGISTRY)

    return Response(
        content=metrics,
        media_type=CONTENT_TYPE_LATEST
    )


# Context manager for timing operations
class timer:
    """Context manager to time operations and record metrics"""

    def __init__(self, operation_name: str = "operation"):
        self.operation_name = operation_name
        self.start_time = None
        self.duration = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.time() - self.start_time
        logger.debug(f"{self.operation_name} took {self.duration:.4f}s")
        return False


if __name__ == "__main__":
    # Demo usage
    print("Metrics Collector Demo")
    print("=" * 50)

    # Record some sample metrics
    collector = MetricsCollector()

    collector.record_request("POST", "/predict", 200, 0.15)
    collector.record_request("POST", "/predict", 200, 0.12)
    collector.record_request("GET", "/health", 200, 0.01)

    collector.record_prediction("APPROVE", "LOW", 0.23, 0.14)
    collector.record_prediction("REJECT", "HIGH", 0.78, 0.15)

    collector.record_model_status("lightgbm", True)
    collector.record_model_status("xgboost", True)
    collector.record_model_status("catboost", True)

    collector.update_system_metrics()

    print("Metrics recorded successfully")
    print(f"Uptime: {collector.get_uptime_seconds():.2f}s")
