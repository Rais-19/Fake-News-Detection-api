from pydantic import BaseModel, Field, field_validator
from typing import Optional


class PredictionRequest(BaseModel):
    """
    Schema for prediction request.
    This defines what data the API expects from users.
    """
    text: str = Field(
        ...,  # Required field
        min_length=10,
        max_length=5000,
        description="The news article text or title to classify",
        examples=["Breaking news: Scientists discover new planet"]
    )
    
    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v):
        """Validate that text is not just whitespace"""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty or just whitespace")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "President announces new policy at press conference today"
            }
        }


class PredictionResponse(BaseModel):
    """
    Schema for prediction response.
    This defines what data the API returns to users.
    """
    prediction: str = Field(
        ...,
        description="Classification result: either 'REAL' or 'FAKE'"
    )
    
    label: int = Field(
        ...,
        description="Numeric label: 0 for FAKE, 1 for REAL"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )
    
    text_length: int = Field(
        ...,
        description="Length of the input text"
    )
    
    message: str = Field(
        ...,
        description="Human-readable message about the prediction"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "FAKE",
                "label": 0,
                "confidence": 0.87,
                "text_length": 45,
                "message": "This news article is classified as FAKE with 87% confidence"
            }
        }


class HealthResponse(BaseModel):
    """
    Schema for health check endpoint response.
    """
    status: str = Field(
        ...,
        description="Service status"
    )
    
    model_loaded: bool = Field(
        ...,
        description="Whether the ML model is loaded"
    )
    
    message: str = Field(
        ...,
        description="Status message"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "message": "Fake News Detection API is running"
            }
        }


class ErrorResponse(BaseModel):
    """
    Schema for error responses.
    """
    error: str = Field(
        ...,
        description="Error type"
    )
    
    message: str = Field(
        ...,
        description="Detailed error message"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Text input cannot be empty"
            }
        }