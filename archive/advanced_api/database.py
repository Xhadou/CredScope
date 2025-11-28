"""Database Integration for Prediction Logging and Audit Trail

Provides SQLite/PostgreSQL integration for storing predictions and audit logs.
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from contextlib import contextmanager
import os

logger = logging.getLogger(__name__)


class PredictionDatabase:
    """Database for storing predictions and audit trail"""

    def __init__(self, db_path: str = "data/predictions.db"):
        """Initialize database

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_database()

    def _ensure_database(self):
        """Ensure database and tables exist"""
        # Create directory if needed
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Create tables
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT UNIQUE,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    api_key_hash TEXT,
                    client_name TEXT,

                    -- Input features (JSON)
                    input_data TEXT,

                    -- Prediction results
                    default_probability REAL,
                    predicted_class INTEGER,
                    decision TEXT,
                    risk_level TEXT,
                    confidence REAL,

                    -- Model info
                    model_version TEXT,
                    prediction_latency_ms REAL,

                    -- Audit
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # SHAP explanations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS shap_explanations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id INTEGER,
                    feature_name TEXT,
                    feature_value REAL,
                    shap_value REAL,
                    rank INTEGER,

                    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
                )
            """)

            # Model performance tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    model_name TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    metadata TEXT
                )
            """)

            # API usage stats
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    api_key_hash TEXT,
                    endpoint TEXT,
                    method TEXT,
                    status_code INTEGER,
                    duration_ms REAL,
                    error_message TEXT
                )
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
                ON predictions(timestamp)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_decision
                ON predictions(decision)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_usage_timestamp
                ON api_usage(timestamp)
            """)

            conn.commit()

        logger.info(f"Database initialized at {self.db_path}")

    @contextmanager
    def get_connection(self):
        """Get database connection with context manager

        Yields:
            sqlite3.Connection: Database connection
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        try:
            yield conn
        finally:
            conn.close()

    def log_prediction(
        self,
        request_id: str,
        input_data: Dict[str, Any],
        prediction_result: Dict[str, Any],
        api_key_hash: Optional[str] = None,
        client_name: Optional[str] = None,
        latency_ms: Optional[float] = None,
        model_version: str = "1.0.0"
    ) -> int:
        """Log a prediction to database

        Args:
            request_id: Unique request identifier
            input_data: Input features dictionary
            prediction_result: Prediction result dictionary
            api_key_hash: Hash of API key (for privacy)
            client_name: Client name
            latency_ms: Prediction latency in milliseconds
            model_version: Model version used

        Returns:
            Prediction ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO predictions (
                    request_id, api_key_hash, client_name,
                    input_data, default_probability, predicted_class,
                    decision, risk_level, confidence,
                    model_version, prediction_latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                request_id,
                api_key_hash,
                client_name,
                json.dumps(input_data),
                prediction_result.get('default_probability'),
                prediction_result.get('predicted_class'),
                prediction_result.get('decision'),
                prediction_result.get('risk_level'),
                prediction_result.get('confidence'),
                model_version,
                latency_ms
            ))

            prediction_id = cursor.lastrowid

            # Log SHAP values if available
            if 'top_features' in prediction_result:
                for rank, feature in enumerate(prediction_result['top_features'], 1):
                    cursor.execute("""
                        INSERT INTO shap_explanations (
                            prediction_id, feature_name, feature_value,
                            shap_value, rank
                        ) VALUES (?, ?, ?, ?, ?)
                    """, (
                        prediction_id,
                        feature.get('feature'),
                        feature.get('feature_value'),
                        feature.get('shap_value'),
                        rank
                    ))

            conn.commit()

        logger.debug(f"Logged prediction {request_id} (ID: {prediction_id})")
        return prediction_id

    def log_api_usage(
        self,
        api_key_hash: Optional[str],
        endpoint: str,
        method: str,
        status_code: int,
        duration_ms: float,
        error_message: Optional[str] = None
    ):
        """Log API usage

        Args:
            api_key_hash: Hash of API key
            endpoint: API endpoint
            method: HTTP method
            status_code: Response status code
            duration_ms: Request duration in milliseconds
            error_message: Error message if any
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO api_usage (
                    api_key_hash, endpoint, method,
                    status_code, duration_ms, error_message
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                api_key_hash,
                endpoint,
                method,
                status_code,
                duration_ms,
                error_message
            ))

            conn.commit()

    def get_predictions(
        self,
        limit: int = 100,
        offset: int = 0,
        decision: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """Get predictions with filtering

        Args:
            limit: Maximum number of results
            offset: Number of results to skip
            decision: Filter by decision (APPROVE/REJECT)
            start_date: Filter by start date
            end_date: Filter by end date

        Returns:
            List of prediction dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM predictions WHERE 1=1"
            params = []

            if decision:
                query += " AND decision = ?"
                params.append(decision)

            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())

            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())

            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)

            results = []
            for row in cursor.fetchall():
                result = dict(row)
                # Parse JSON fields
                if result['input_data']:
                    result['input_data'] = json.loads(result['input_data'])
                results.append(result)

            return results

    def get_statistics(self, days: int = 7) -> Dict:
        """Get prediction statistics

        Args:
            days: Number of days to analyze

        Returns:
            Statistics dictionary
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            # Total predictions
            cursor.execute("""
                SELECT COUNT(*) as total
                FROM predictions
                WHERE timestamp >= datetime('now', '-' || ? || ' days')
            """, (days,))
            stats['total_predictions'] = cursor.fetchone()['total']

            # Decisions breakdown
            cursor.execute("""
                SELECT decision, COUNT(*) as count
                FROM predictions
                WHERE timestamp >= datetime('now', '-' || ? || ' days')
                GROUP BY decision
            """, (days,))
            stats['decisions'] = {row['decision']: row['count'] for row in cursor.fetchall()}

            # Risk level distribution
            cursor.execute("""
                SELECT risk_level, COUNT(*) as count
                FROM predictions
                WHERE timestamp >= datetime('now', '-' || ? || ' days')
                GROUP BY risk_level
            """, (days,))
            stats['risk_levels'] = {row['risk_level']: row['count'] for row in cursor.fetchall()}

            # Average probability by decision
            cursor.execute("""
                SELECT decision, AVG(default_probability) as avg_prob
                FROM predictions
                WHERE timestamp >= datetime('now', '-' || ? || ' days')
                GROUP BY decision
            """, (days,))
            stats['avg_probability'] = {
                row['decision']: row['avg_prob'] for row in cursor.fetchall()
            }

            # Average latency
            cursor.execute("""
                SELECT AVG(prediction_latency_ms) as avg_latency
                FROM predictions
                WHERE timestamp >= datetime('now', '-' || ? || ' days')
                AND prediction_latency_ms IS NOT NULL
            """, (days,))
            result = cursor.fetchone()
            stats['avg_latency_ms'] = result['avg_latency'] if result['avg_latency'] else 0

            return stats

    def get_top_features(self, decision: Optional[str] = None, limit: int = 20) -> List[Dict]:
        """Get most important features across predictions

        Args:
            decision: Filter by decision (optional)
            limit: Number of features to return

        Returns:
            List of features with average SHAP values
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT
                    s.feature_name,
                    AVG(ABS(s.shap_value)) as avg_abs_shap,
                    COUNT(*) as frequency
                FROM shap_explanations s
                JOIN predictions p ON s.prediction_id = p.id
            """

            params = []
            if decision:
                query += " WHERE p.decision = ?"
                params.append(decision)

            query += """
                GROUP BY s.feature_name
                ORDER BY avg_abs_shap DESC
                LIMIT ?
            """
            params.append(limit)

            cursor.execute(query, params)

            return [dict(row) for row in cursor.fetchall()]


# Global database instance
_db_instance = None


def get_database() -> PredictionDatabase:
    """Get or create global database instance

    Returns:
        PredictionDatabase instance
    """
    global _db_instance
    if _db_instance is None:
        db_path = os.getenv("DB_PATH", "data/predictions.db")
        _db_instance = PredictionDatabase(db_path)
    return _db_instance


if __name__ == "__main__":
    # Demo usage
    print("Prediction Database Demo")
    print("=" * 50)

    db = PredictionDatabase("test_predictions.db")

    # Log a prediction
    prediction_id = db.log_prediction(
        request_id="test-123",
        input_data={"AMT_INCOME_TOTAL": 180000, "AMT_CREDIT": 500000},
        prediction_result={
            "default_probability": 0.23,
            "predicted_class": 0,
            "decision": "APPROVE",
            "risk_level": "LOW",
            "confidence": 0.54,
            "top_features": [
                {"feature": "INT_EXT_AVG", "feature_value": 0.65, "shap_value": -0.15},
                {"feature": "DAYS_BIRTH", "feature_value": -15000, "shap_value": -0.10}
            ]
        },
        client_name="Test Client",
        latency_ms=142.5
    )

    print(f"Logged prediction with ID: {prediction_id}")

    # Get statistics
    stats = db.get_statistics(days=30)
    print(f"\nStatistics: {json.dumps(stats, indent=2)}")

    # Get top features
    top_features = db.get_top_features(limit=5)
    print(f"\nTop features: {json.dumps(top_features, indent=2)}")

    # Cleanup
    Path("test_predictions.db").unlink()
