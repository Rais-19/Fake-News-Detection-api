from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ErrorResponse
)
from services.prediction_service import PredictionService


# App initialization

app = FastAPI(
    title="Fake News Detection API",
    description="Detect whether a news article is REAL or FAKE",
    version="1.0.0"
)

prediction_service = PredictionService()


# Root endpoint

@app.get("/")
def root():
    return {
        "message": "Welcome to Fake News Detection API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }


# Health check

@app.get("/health", response_model=HealthResponse)
def health_check():
    model_loaded = (
        prediction_service.model is not None and
        prediction_service.vectorizer is not None
    )

    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        message="API is running"
    )


# Prediction endpoint

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # The service returns a dict, not a tuple!
        result = prediction_service.predict(request.text)
        
        return PredictionResponse(
            prediction=result['prediction'],  # 'REAL' or 'FAKE'
            label=result['label'],            # 0 or 1
            confidence=result['confidence'],  # 0.0 to 1.0
            text_length=len(request.text),
            message=f"News classified as {result['prediction']} with {int(result['confidence']*100)}% confidence"
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


# Global exception handler

@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException
):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="APIError",
            message=str(exc.detail)
        ).model_dump()
    )