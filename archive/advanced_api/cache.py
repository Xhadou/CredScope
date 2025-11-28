"""Caching Layer for Performance Optimization

Provides in-memory and Redis caching for predictions and features.
"""

import hashlib
import json
import pickle
import time
import logging
from typing import Any, Optional, Callable
from functools import wraps
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)


class LRUCache:
    """Thread-safe LRU (Least Recently Used) Cache"""

    def __init__(self, max_size: int = 1000):
        """Initialize LRU cache

        Args:
            max_size: Maximum number of items to cache
        """
        self.cache = OrderedDict()
        self.max_size = max_size
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]['value']
            else:
                self.misses += 1
                return None

    def put(self, key: str, value: Any, ttl: int = 3600):
        """Put item in cache

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        with self.lock:
            # Remove oldest item if cache is full
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)

            # Add new item
            self.cache[key] = {
                'value': value,
                'expires_at': time.time() + ttl
            }

            # Move to end (most recently used)
            self.cache.move_to_end(key)

    def invalidate(self, key: str):
        """Remove item from cache

        Args:
            key: Cache key
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]

    def clear(self):
        """Clear all cached items"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    def cleanup_expired(self):
        """Remove expired items from cache"""
        with self.lock:
            current_time = time.time()
            expired_keys = [
                key for key, data in self.cache.items()
                if data['expires_at'] < current_time
            ]

            for key in expired_keys:
                del self.cache[key]

            return len(expired_keys)

    def get_stats(self) -> dict:
        """Get cache statistics

        Returns:
            Dictionary with cache stats
        """
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0

            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate
            }


class PredictionCache:
    """Cache for prediction results"""

    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        """Initialize prediction cache

        Args:
            max_size: Maximum cache size
            ttl: Default time to live in seconds
        """
        self.cache = LRUCache(max_size)
        self.ttl = ttl

    def _generate_key(self, input_data: dict) -> str:
        """Generate cache key from input data

        Args:
            input_data: Input features dictionary

        Returns:
            Cache key (hash)
        """
        # Sort dictionary for consistent hashing
        sorted_data = json.dumps(input_data, sort_keys=True)
        return hashlib.sha256(sorted_data.encode()).hexdigest()

    def get_prediction(self, input_data: dict) -> Optional[dict]:
        """Get cached prediction

        Args:
            input_data: Input features

        Returns:
            Cached prediction or None
        """
        key = self._generate_key(input_data)
        result = self.cache.get(key)

        if result:
            logger.debug(f"Cache HIT for key {key[:8]}...")
        else:
            logger.debug(f"Cache MISS for key {key[:8]}...")

        return result

    def cache_prediction(self, input_data: dict, prediction: dict):
        """Cache prediction result

        Args:
            input_data: Input features
            prediction: Prediction result
        """
        key = self._generate_key(input_data)
        self.cache.put(key, prediction, ttl=self.ttl)
        logger.debug(f"Cached prediction for key {key[:8]}...")

    def clear(self):
        """Clear all cached predictions"""
        self.cache.clear()
        logger.info("Prediction cache cleared")

    def get_stats(self) -> dict:
        """Get cache statistics"""
        return self.cache.get_stats()


class FeatureCache:
    """Cache for engineered features"""

    def __init__(self, max_size: int = 5000):
        """Initialize feature cache

        Args:
            max_size: Maximum cache size
        """
        self.cache = LRUCache(max_size)

    def _generate_key(self, applicant_id: str, feature_version: str = "v1") -> str:
        """Generate cache key

        Args:
            applicant_id: Unique applicant identifier
            feature_version: Feature engineering version

        Returns:
            Cache key
        """
        return f"features:{feature_version}:{applicant_id}"

    def get_features(self, applicant_id: str) -> Optional[dict]:
        """Get cached features

        Args:
            applicant_id: Applicant identifier

        Returns:
            Cached features or None
        """
        key = self._generate_key(applicant_id)
        return self.cache.get(key)

    def cache_features(self, applicant_id: str, features: dict):
        """Cache engineered features

        Args:
            applicant_id: Applicant identifier
            features: Engineered features
        """
        key = self._generate_key(applicant_id)
        self.cache.put(key, features, ttl=7200)  # 2 hours


# Global cache instances
prediction_cache = PredictionCache(max_size=10000, ttl=1800)  # 30 minutes
feature_cache = FeatureCache(max_size=5000)


def cached_prediction(func: Callable) -> Callable:
    """Decorator to cache prediction results

    Args:
        func: Function to cache

    Returns:
        Wrapped function with caching
    """
    @wraps(func)
    def wrapper(input_data: dict, *args, **kwargs):
        # Try to get from cache
        cached_result = prediction_cache.get_prediction(input_data)
        if cached_result is not None:
            # Add cache indicator
            cached_result['from_cache'] = True
            return cached_result

        # Compute result
        result = func(input_data, *args, **kwargs)

        # Cache result
        if result is not None:
            result['from_cache'] = False
            prediction_cache.cache_prediction(input_data, result)

        return result

    return wrapper


def cache_stats() -> dict:
    """Get combined cache statistics

    Returns:
        Dictionary with all cache stats
    """
    return {
        'prediction_cache': prediction_cache.get_stats(),
        'feature_cache': feature_cache.get_stats()
    }


def clear_all_caches():
    """Clear all caches"""
    prediction_cache.clear()
    feature_cache.clear()
    logger.info("All caches cleared")


if __name__ == "__main__":
    # Demo usage
    print("Caching System Demo")
    print("=" * 50)

    # Test prediction cache
    cache = PredictionCache(max_size=100, ttl=60)

    input_data = {
        "AMT_INCOME_TOTAL": 180000,
        "AMT_CREDIT": 500000
    }

    prediction = {
        "default_probability": 0.23,
        "decision": "APPROVE"
    }

    # First request - cache miss
    result = cache.get_prediction(input_data)
    print(f"First request (should be None): {result}")

    # Cache the result
    cache.cache_prediction(input_data, prediction)

    # Second request - cache hit
    result = cache.get_prediction(input_data)
    print(f"Second request (should be cached): {result}")

    # Get stats
    stats = cache.get_stats()
    print(f"\nCache stats: {json.dumps(stats, indent=2)}")
    print(f"Hit rate: {stats['hit_rate']:.1%}")
