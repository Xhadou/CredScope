"""FastAPI REST API for CredScope Credit Risk Assessment

This module provides HTTP endpoints for credit risk predictions.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import numpy as np

from .predictor import get_predictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="CredScope API",
    description="Alternative Credit Risk Assessment API using behavioral and alternative data sources",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response validation
class ApplicantInput(BaseModel):
    """Input model for loan applicant data"""

    # === REQUIRED FIELDS ===
    AMT_INCOME_TOTAL: float = Field(..., description="Total annual income of applicant", gt=0)
    AMT_CREDIT: float = Field(..., description="Credit amount of the loan", gt=0)
    DAYS_BIRTH: int = Field(..., description="Age in days (negative value)", lt=0)

    # === CREDIT DETAILS ===
    AMT_ANNUITY: Optional[float] = Field(None, description="Loan annuity payment", ge=0)
    AMT_GOODS_PRICE: Optional[float] = Field(None, description="Price of goods for consumer loans", ge=0)
    NAME_CONTRACT_TYPE: Optional[str] = Field("Cash loans", description="Contract type: Cash loans or Revolving loans")

    # === PERSONAL INFO ===
    CODE_GENDER: Optional[int] = Field(0, description="Gender code (0=Female, 1=Male)", ge=0, le=1)
    CNT_CHILDREN: Optional[int] = Field(0, description="Number of children", ge=0)
    CNT_FAM_MEMBERS: Optional[float] = Field(2.0, description="Number of family members", ge=1)
    NAME_FAMILY_STATUS: Optional[str] = Field("Married", description="Family status")

    # === EDUCATION ===
    NAME_EDUCATION_TYPE: Optional[str] = Field(
        "Secondary / secondary special", 
        description="Education level: Lower secondary, Secondary / secondary special, Incomplete higher, Higher education, Academic degree"
    )

    # === HOUSING & ASSETS ===
    NAME_HOUSING_TYPE: Optional[str] = Field(
        "House / apartment",
        description="Housing type: House / apartment, Rented apartment, With parents, Municipal apartment, Office apartment, Co-op apartment"
    )
    FLAG_OWN_CAR: Optional[str] = Field("N", description="Owns a car: Y or N")
    FLAG_OWN_REALTY: Optional[str] = Field("Y", description="Owns real estate: Y or N")
    OWN_CAR_AGE: Optional[float] = Field(None, description="Age of car in years if owned")

    # === EMPLOYMENT ===
    DAYS_EMPLOYED: Optional[int] = Field(None, description="Employment duration in days (negative if employed)")
    NAME_INCOME_TYPE: Optional[str] = Field(
        "Working",
        description="Income type: Working, Commercial associate, Pensioner, State servant, Student, Unemployed"
    )
    OCCUPATION_TYPE: Optional[str] = Field(None, description="Occupation type")
    ORGANIZATION_TYPE: Optional[str] = Field(None, description="Organization type of employer")

    # === CONTACT INFO ===
    FLAG_MOBIL: Optional[int] = Field(1, description="Has mobile phone: 0 or 1", ge=0, le=1)
    FLAG_PHONE: Optional[int] = Field(0, description="Has home phone: 0 or 1", ge=0, le=1)
    FLAG_EMAIL: Optional[int] = Field(0, description="Has email: 0 or 1", ge=0, le=1)
    FLAG_WORK_PHONE: Optional[int] = Field(0, description="Has work phone: 0 or 1", ge=0, le=1)

    # === EXTERNAL CREDIT SCORES (Most Important Features) ===
    EXT_SOURCE_1: Optional[float] = Field(None, description="External credit score 1 (normalized 0-1)", ge=0, le=1)
    EXT_SOURCE_2: Optional[float] = Field(None, description="External credit score 2 (normalized 0-1)", ge=0, le=1)
    EXT_SOURCE_3: Optional[float] = Field(None, description="External credit score 3 (normalized 0-1)", ge=0, le=1)

    # === REGION & ADDRESS ===
    REGION_POPULATION_RELATIVE: Optional[float] = Field(None, description="Normalized population of region", ge=0, le=1)
    REGION_RATING_CLIENT: Optional[int] = Field(None, description="Region rating (1-3)", ge=1, le=3)

    # === DOCUMENT FLAGS ===
    FLAG_DOCUMENT_3: Optional[int] = Field(1, description="Provided document 3: 0 or 1", ge=0, le=1)
    FLAG_DOCUMENT_6: Optional[int] = Field(0, description="Provided document 6: 0 or 1", ge=0, le=1)
    FLAG_DOCUMENT_8: Optional[int] = Field(0, description="Provided document 8: 0 or 1", ge=0, le=1)

    # Additional features (optional - for advanced users)
    additional_features: Optional[Dict[str, float]] = Field(
        default_factory=dict,
        description="Additional engineered features from bureau, installments, etc."
    )

    class Config:
        schema_extra = {
            "example": {
                "AMT_INCOME_TOTAL": 180000,
                "AMT_CREDIT": 500000,
                "AMT_ANNUITY": 25000,
                "AMT_GOODS_PRICE": 450000,
                "DAYS_BIRTH": -15000,
                "DAYS_EMPLOYED": -3000,
                "CODE_GENDER": 1,
                "CNT_CHILDREN": 0,
                "CNT_FAM_MEMBERS": 2,
                "NAME_EDUCATION_TYPE": "Higher education",
                "NAME_HOUSING_TYPE": "House / apartment",
                "NAME_INCOME_TYPE": "Working",
                "FLAG_OWN_CAR": "Y",
                "FLAG_OWN_REALTY": "Y",
                "EXT_SOURCE_1": 0.65,
                "EXT_SOURCE_2": 0.72,
                "EXT_SOURCE_3": 0.58
            }
        }


class PredictionResponse(BaseModel):
    """Response model for prediction"""

    default_probability: float = Field(..., description="Probability of default (0-1)")
    predicted_class: int = Field(..., description="Predicted class (0=repay, 1=default)")
    decision: str = Field(..., description="Lending decision (APPROVE/REJECT)")
    confidence: float = Field(..., description="Confidence score (0-1)")
    risk_level: str = Field(..., description="Risk category")
    timestamp: str = Field(..., description="Prediction timestamp")


class ExplanationResponse(BaseModel):
    """Response model with SHAP explanation"""

    default_probability: float
    predicted_class: int
    decision: str
    confidence: float
    risk_level: str
    top_features: List[Dict[str, Any]] = Field(..., description="Top contributing features")
    timestamp: str


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""

    applicants: List[ApplicantInput]


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""

    predictions: List[PredictionResponse]
    total_count: int
    approved_count: int
    rejected_count: int


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    timestamp: str
    models_loaded: bool
    version: str


# Global predictor (lazy loaded)
predictor = None


def get_loaded_predictor():
    """Get or initialize the predictor"""
    global predictor
    if predictor is None:
        try:
            predictor = get_predictor(models_dir="models")
            logger.info("✅ Predictor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model loading failed: {str(e)}"
            )
    return predictor


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting CredScope API...")
    try:
        get_loaded_predictor()
        logger.info("✅ API ready to serve requests")
    except Exception as e:
        logger.error(f"Startup failed: {e}")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "CredScope Credit Risk Assessment API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    try:
        pred = get_loaded_predictor()
        models_loaded = pred.loaded
    except:
        models_loaded = False

    return HealthResponse(
        status="healthy" if models_loaded else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        models_loaded=models_loaded,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(applicant: ApplicantInput):
    """
    Make credit risk prediction for a single applicant

    This endpoint evaluates an applicant's creditworthiness using alternative data
    and returns a default probability with a lending decision.

    **Returns:**
    - default_probability: Probability of default (0-1)
    - decision: APPROVE or REJECT
    - risk_level: Risk category (VERY_LOW to VERY_HIGH)
    """
    try:
        pred = get_loaded_predictor()

        # Prepare input data
        input_data = applicant.dict(exclude={'additional_features'})

        # Add additional features if provided
        if applicant.additional_features:
            input_data.update(applicant.additional_features)

        # Make prediction
        result = pred.predict_single(input_data)

        # Add timestamp
        result['timestamp'] = datetime.utcnow().isoformat()

        return PredictionResponse(**result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/explain", response_model=ExplanationResponse, tags=["Predictions"])
async def predict_with_explanation(applicant: ApplicantInput):
    """
    Make prediction with SHAP explanation

    Returns prediction along with the top features that influenced the decision,
    helping understand why an applicant was approved or rejected.

    **Returns:**
    - Prediction results
    - top_features: List of most influential features with SHAP values
    """
    try:
        pred = get_loaded_predictor()

        # Prepare input data
        input_data = applicant.dict(exclude={'additional_features'})
        if applicant.additional_features:
            input_data.update(applicant.additional_features)

        # Get explanation
        result = pred.explain_prediction(input_data, top_n=10)

        # Add timestamp
        result['timestamp'] = datetime.utcnow().isoformat()

        return ExplanationResponse(**result)

    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Make predictions for multiple applicants

    Efficient batch processing for scoring multiple loan applications at once.

    **Returns:**
    - predictions: List of prediction results
    - Summary statistics
    """
    try:
        pred = get_loaded_predictor()

        predictions = []
        for applicant in request.applicants:
            # Prepare input data
            input_data = applicant.dict(exclude={'additional_features'})
            if applicant.additional_features:
                input_data.update(applicant.additional_features)

            # Make prediction
            result = pred.predict_single(input_data)
            result['timestamp'] = datetime.utcnow().isoformat()
            predictions.append(PredictionResponse(**result))

        # Calculate summary stats
        total_count = len(predictions)
        approved_count = sum(1 for p in predictions if p.decision == "APPROVE")
        rejected_count = total_count - approved_count

        return BatchPredictionResponse(
            predictions=predictions,
            total_count=total_count,
            approved_count=approved_count,
            rejected_count=rejected_count
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/models/info", tags=["Models"])
async def model_info():
    """Get information about loaded models"""
    try:
        pred = get_loaded_predictor()

        return {
            "models_loaded": pred.loaded,
            "lightgbm": pred.lgb_model is not None,
            "xgboost": pred.xgb_model is not None,
            "catboost": pred.catb_model is not None,
            "meta_model": pred.meta_model is not None,
            "num_features": len(pred.feature_names) if pred.feature_names else None
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
