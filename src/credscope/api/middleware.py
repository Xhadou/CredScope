"""API Middleware for logging, monitoring, and rate limiting

Provides middleware components for request/response processing.
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import json

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all API requests and responses"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Start timer
        start_time = time.time()

        # Log request
        logger.info(
            f"Request started",
            extra={
                "method": request.method,
                "url": str(request.url),
                "client": request.client.host if request.client else "unknown",
            }
        )

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Log response
        logger.info(
            f"Request completed",
            extra={
                "method": request.method,
                "url": str(request.url),
                "status_code": response.status_code,
                "duration_ms": round(duration * 1000, 2),
            }
        )

        # Add custom headers
        response.headers["X-Process-Time"] = str(duration)

        return response


class RateLimitHeaderMiddleware(BaseHTTPMiddleware):
    """Add rate limit headers to responses"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Add rate limit info if available (set by verify_api_key dependency)
        if hasattr(request.state, "rate_limit_info"):
            info = request.state.rate_limit_info
            response.headers["X-RateLimit-Limit"] = str(info.get("limit", ""))
            response.headers["X-RateLimit-Remaining"] = str(info.get("remaining", ""))
            response.headers["X-RateLimit-Reset"] = str(info.get("reset_in_seconds", ""))

        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        import uuid

        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Process request
        response = await call_next(request)

        # Add request ID to response
        response.headers["X-Request-ID"] = request_id

        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            # Log error
            logger.error(
                f"Unhandled exception",
                exc_info=exc,
                extra={
                    "method": request.method,
                    "url": str(request.url),
                    "request_id": getattr(request.state, "request_id", "unknown"),
                }
            )

            # Return generic error response
            return Response(
                content=json.dumps({
                    "detail": "Internal server error",
                    "request_id": getattr(request.state, "request_id", "unknown"),
                    "type": "internal_error"
                }),
                status_code=500,
                media_type="application/json"
            )


def setup_middleware(app: ASGIApp):
    """Setup all middleware for the application

    Args:
        app: FastAPI application instance
    """
    # Add middleware in reverse order (last added = first executed)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(RateLimitHeaderMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    logger.info("Middleware setup complete")
