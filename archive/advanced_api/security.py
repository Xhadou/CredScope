"""API Security - Authentication and Rate Limiting

Provides API key authentication and rate limiting for production use.
"""

import hashlib
import secrets
import time
from typing import Optional, Dict
from datetime import datetime, timedelta
from collections import defaultdict
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
import logging

logger = logging.getLogger(__name__)

# API Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class APIKeyManager:
    """Manage API keys and validation"""

    def __init__(self):
        # In production, store these in a database or secrets manager
        self.valid_keys = {
            # Format: hashed_key: {"name": "client_name", "tier": "free|pro|enterprise", "created": datetime}
        }
        self._init_demo_keys()

    def _init_demo_keys(self):
        """Initialize demo API keys for testing"""
        # Demo key: demo_key_12345 (DO NOT USE IN PRODUCTION)
        demo_key = "demo_key_12345"
        hashed = self._hash_key(demo_key)
        self.valid_keys[hashed] = {
            "name": "Demo Client",
            "tier": "free",
            "created": datetime.utcnow(),
            "requests_limit": 100,  # per hour
        }

        # Admin key: admin_key_secret (DO NOT USE IN PRODUCTION)
        admin_key = "admin_key_secret"
        hashed = self._hash_key(admin_key)
        self.valid_keys[hashed] = {
            "name": "Admin",
            "tier": "enterprise",
            "created": datetime.utcnow(),
            "requests_limit": 10000,  # per hour
        }

    def _hash_key(self, api_key: str) -> str:
        """Hash API key for secure storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def validate_key(self, api_key: str) -> Optional[Dict]:
        """Validate API key and return client info

        Args:
            api_key: API key to validate

        Returns:
            Client info dict if valid, None otherwise
        """
        if not api_key:
            return None

        hashed = self._hash_key(api_key)
        return self.valid_keys.get(hashed)

    def generate_key(self, client_name: str, tier: str = "free") -> str:
        """Generate new API key

        Args:
            client_name: Name of the client
            tier: Service tier (free, pro, enterprise)

        Returns:
            Generated API key
        """
        # Generate secure random key
        raw_key = f"cred_{secrets.token_urlsafe(32)}"
        hashed = self._hash_key(raw_key)

        # Store key info
        limits = {
            "free": 100,
            "pro": 1000,
            "enterprise": 10000
        }

        self.valid_keys[hashed] = {
            "name": client_name,
            "tier": tier,
            "created": datetime.utcnow(),
            "requests_limit": limits.get(tier, 100)
        }

        logger.info(f"Generated new API key for {client_name} (tier: {tier})")

        return raw_key


class RateLimiter:
    """Token bucket rate limiter"""

    def __init__(self):
        # Store: {api_key: {"tokens": int, "last_update": float, "requests": int}}
        self.buckets: Dict[str, Dict] = defaultdict(lambda: {
            "tokens": 0,
            "last_update": time.time(),
            "requests": 0,
            "window_start": time.time()
        })

    def is_allowed(
        self,
        api_key: str,
        rate_limit: int = 100,
        window: int = 3600  # 1 hour in seconds
    ) -> tuple[bool, Dict]:
        """Check if request is allowed under rate limit

        Args:
            api_key: API key making the request
            rate_limit: Maximum requests per window
            window: Time window in seconds

        Returns:
            Tuple of (is_allowed, info_dict)
        """
        bucket = self.buckets[api_key]
        current_time = time.time()

        # Reset window if expired
        if current_time - bucket["window_start"] > window:
            bucket["requests"] = 0
            bucket["window_start"] = current_time

        # Check limit
        if bucket["requests"] >= rate_limit:
            time_until_reset = window - (current_time - bucket["window_start"])
            return False, {
                "allowed": False,
                "limit": rate_limit,
                "remaining": 0,
                "reset_in_seconds": int(time_until_reset)
            }

        # Allow request
        bucket["requests"] += 1

        return True, {
            "allowed": True,
            "limit": rate_limit,
            "remaining": rate_limit - bucket["requests"],
            "reset_in_seconds": int(window - (current_time - bucket["window_start"]))
        }

    def get_stats(self, api_key: str) -> Dict:
        """Get rate limit stats for an API key"""
        bucket = self.buckets[api_key]
        return {
            "requests_made": bucket["requests"],
            "window_start": datetime.fromtimestamp(bucket["window_start"]).isoformat()
        }


# Global instances
api_key_manager = APIKeyManager()
rate_limiter = RateLimiter()


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header)
) -> Dict:
    """Dependency to verify API key

    Args:
        api_key: API key from request header

    Returns:
        Client info dict

    Raises:
        HTTPException: If API key is invalid or rate limited
    """
    # For development, allow requests without API key
    # Set REQUIRE_API_KEY=true in production
    import os
    if os.getenv("REQUIRE_API_KEY", "false").lower() != "true":
        if not api_key:
            return {
                "name": "Anonymous",
                "tier": "free",
                "requests_limit": 100
            }

    # Validate API key
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Include X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    client_info = api_key_manager.validate_key(api_key)
    if not client_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    # Check rate limit
    allowed, rate_info = rate_limiter.is_allowed(
        api_key,
        rate_limit=client_info.get("requests_limit", 100)
    )

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Resets in {rate_info['reset_in_seconds']} seconds.",
            headers={
                "X-RateLimit-Limit": str(rate_info["limit"]),
                "X-RateLimit-Remaining": str(rate_info["remaining"]),
                "X-RateLimit-Reset": str(rate_info["reset_in_seconds"]),
                "Retry-After": str(rate_info["reset_in_seconds"])
            }
        )

    # Add rate limit info to response headers (handled by middleware)
    client_info["rate_limit_info"] = rate_info

    return client_info


def get_api_key_stats() -> Dict:
    """Get statistics about API key usage

    Returns:
        Dictionary with API key usage stats
    """
    total_keys = len(api_key_manager.valid_keys)
    tier_counts = defaultdict(int)

    for key_info in api_key_manager.valid_keys.values():
        tier_counts[key_info["tier"]] += 1

    return {
        "total_api_keys": total_keys,
        "keys_by_tier": dict(tier_counts),
        "active_requests": len(rate_limiter.buckets)
    }


if __name__ == "__main__":
    # Demo usage
    print("API Security Module")
    print("=" * 50)

    # Generate a new key
    manager = APIKeyManager()
    new_key = manager.generate_key("Test Client", "pro")
    print(f"Generated API key: {new_key}")

    # Validate key
    info = manager.validate_key(new_key)
    print(f"Key info: {info}")

    # Test rate limiter
    limiter = RateLimiter()
    for i in range(5):
        allowed, info = limiter.is_allowed(new_key, rate_limit=3, window=60)
        print(f"Request {i+1}: Allowed={allowed}, Info={info}")
