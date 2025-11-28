"""Enhanced FastAPI REST API with Full Production Features

Includes authentication, rate limiting, monitoring, caching, and database logging.
"""

from fastapi import FastAPI, HTTPException, status, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import time
import uuid

from .predictor import get_predictor
from .security import verify_api_key, api_key_manager, rate_limiter
from .middleware import setup_middleware
from .monitoring import metrics_collector, get_metrics, timer
from .database import get_database
from .cache import prediction_cache, cache_stats, cached_prediction
from ..business.metrics import ROICalculator, BusinessMetricsCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="CredScope API v2",
    description="Production-Grade Alternative Credit Risk Assessment API with Authentication, Monitoring, and Caching",
    version="2.0.0",
    docs_url="/api/v2/docs",
    redoc_url="/api/v2/redoc"
)

# Setup middleware
setup_middleware(app)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class ApplicantInput(BaseModel):
    """Input model for loan applicant"""
    AMT_INCOME_TOTAL: float = Field(..., gt=0)
    AMT_CREDIT: float = Field(..., gt=0)
    AMT_ANNUITY: Optional[float] = Field(None, ge=0)
    AMT_GOODS_PRICE: Optional[float] = Field(None, ge=0)
    DAYS_BIRTH: int = Field(..., lt=0)
    DAYS_EMPLOYED: Optional[int] = None
    CODE_GENDER: Optional[int] = Field(0, ge=0, le=1)
    EXT_SOURCE_1: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_2: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_3: Optional[float] = Field(None, ge=0, le=1)
    additional_features: Optional[Dict[str, float]] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    """Enhanced prediction response"""
    default_probability: float
    predicted_class: int
    decision: str
    confidence: float
    risk_level: str
    timestamp: str
    request_id: str
    from_cache: bool = False
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str
    models_loaded: bool
    cache_stats: Dict
    uptime_seconds: float


# Global predictor
predictor = None
database = None
roi_calculator = None


def get_loaded_predictor():
    """Get or initialize predictor"""
    global predictor
    if predictor is None:
        predictor = get_predictor(models_dir="models")
    return predictor


def get_db():
    """Get database instance"""
    global database
    if database is None:
        database = get_database()
    return database


def get_roi_calc():
    """Get ROI calculator"""
    global roi_calculator
    if roi_calculator is None:
        roi_calculator = ROICalculator()
    return roi_calculator


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting CredScope API v2...")

    # Load models
    try:
        pred = get_loaded_predictor()
        metrics_collector.record_model_status("lightgbm", pred.lgb_model is not None)
        metrics_collector.record_model_status("xgboost", pred.xgb_model is not None)
        metrics_collector.record_model_status("catboost", pred.catb_model is not None)
        logger.info("✅ Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")

    # Initialize database
    try:
        get_db()
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

    logger.info("✅ API ready to serve requests")


@app.get("/api/v2/", tags=["General"])
async def root():
    """API root endpoint"""
    return {
        "name": "CredScope API v2",
        "version": "2.0.0",
        "description": "Production-grade credit risk assessment with authentication, monitoring, and caching",
        "endpoints": {
            "docs": "/api/v2/docs",
            "health": "/api/v2/health",
            "predict": "/api/v2/predict",
            "metrics": "/api/v2/metrics",
            "stats": "/api/v2/stats"
        }
    }


@app.get("/api/v2/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Comprehensive health check"""
    try:
        pred = get_loaded_predictor()
        models_loaded = pred.loaded
    except:
        models_loaded = False

    return HealthResponse(
        status="healthy" if models_loaded else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        version="2.0.0",
        models_loaded=models_loaded,
        cache_stats=cache_stats(),
        uptime_seconds=metrics_collector.get_uptime_seconds()
    )


@app.post("/api/v2/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(
    request: Request,
    applicant: ApplicantInput,
    client_info: Dict = Depends(verify_api_key)
):
    """Make credit risk prediction with full production features

    - ✅ API key authentication
    - ✅ Rate limiting
    - ✅ Response caching
    - ✅ Database logging
    - ✅ Prometheus metrics
    - ✅ Performance monitoring
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        # Get predictor
        pred = get_loaded_predictor()

        # Prepare input
        input_data = applicant.dict(exclude={'additional_features'})
        if applicant.additional_features:
            input_data.update(applicant.additional_features)

        # Check cache first
        with timer("prediction"):
            cached_result = prediction_cache.get_prediction(input_data)

            if cached_result:
                # Cache hit
                result = cached_result
                result['from_cache'] = True
                logger.info(f"Cache HIT for request {request_id}")
            else:
                # Cache miss - compute prediction
                result = pred.predict_single(input_data)
                result['from_cache'] = False

                # Cache the result
                prediction_cache.cache_prediction(input_data, result)
                logger.info(f"Cache MISS for request {request_id}")

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Add metadata
        result['timestamp'] = datetime.utcnow().isoformat()
        result['request_id'] = request_id
        result['processing_time_ms'] = processing_time_ms

        # Record metrics
        metrics_collector.record_prediction(
            decision=result['decision'],
            risk_level=result['risk_level'],
            probability=result['default_probability'],
            latency=processing_time_ms / 1000
        )

        # Log to database
        try:
            db = get_db()
            db.log_prediction(
                request_id=request_id,
                input_data=input_data,
                prediction_result=result,
                api_key_hash=client_info.get('name'),
                client_name=client_info.get('name'),
                latency_ms=processing_time_ms
            )
        except Exception as e:
            logger.error(f"Database logging failed: {e}")

        return PredictionResponse(**result)

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        metrics_collector.record_prediction_error(type(e).__name__)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/api/v2/predict/batch", tags=["Predictions"])
async def predict_batch(
    applicants: List[ApplicantInput],
    client_info: Dict = Depends(verify_api_key)
):
    """Batch predictions with monitoring"""
    start_time = time.time()

    results = []
    for applicant in applicants:
        try:
            # Reuse single prediction endpoint logic
            input_data = applicant.dict(exclude={'additional_features'})
            if applicant.additional_features:
                input_data.update(applicant.additional_features)

            pred = get_loaded_predictor()
            result = pred.predict_single(input_data)
            results.append(result)

        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            results.append({'error': str(e)})

    # Calculate business metrics
    roi_calc = get_roi_calc()
    portfolio_metrics = roi_calc.calculate_portfolio_metrics(results)

    processing_time = time.time() - start_time

    return {
        'predictions': results,
        'portfolio_metrics': portfolio_metrics,
        'total_count': len(applicants),
        'successful_count': len([r for r in results if 'error' not in r]),
        'processing_time_seconds': processing_time
    }


@app.get("/api/v2/metrics", tags=["Monitoring"])
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    return get_metrics()


@app.get("/api/v2/stats", tags=["Monitoring"])
async def statistics(
    client_info: Dict = Depends(verify_api_key)
):
    """Get prediction statistics"""
    try:
        db = get_db()
        stats = db.get_statistics(days=7)

        return {
            'period_days': 7,
            'statistics': stats,
            'cache_stats': cache_stats(),
            'uptime_seconds': metrics_collector.get_uptime_seconds()
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )


@app.get("/api/v2/top-features", tags=["Analytics"])
async def top_features(
    decision: Optional[str] = None,
    limit: int = 20,
    client_info: Dict = Depends(verify_api_key)
):
    """Get most important features from historical predictions"""
    try:
        db = get_db()
        features = db.get_top_features(decision=decision, limit=limit)

        return {
            'features': features,
            'decision_filter': decision,
            'count': len(features)
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get features: {str(e)}"
        )


@app.delete("/api/v2/cache/clear", tags=["Admin"])
async def clear_cache(client_info: Dict = Depends(verify_api_key)):
    """Clear prediction cache (admin only)"""
    if client_info.get('tier') != 'enterprise':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )

    from .cache import clear_all_caches
    clear_all_caches()

    return {
        'status': 'success',
        'message': 'All caches cleared',
        'timestamp': datetime.utcnow().isoformat()
    }


@app.get("/api/v2/api-keys/stats", tags=["Admin"])
async def api_key_statistics(client_info: Dict = Depends(verify_api_key)):
    """Get API key usage statistics"""
    if client_info.get('tier') != 'enterprise':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )

    from .security import get_api_key_stats
    return get_api_key_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
