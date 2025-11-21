"""
Pydantic schemas for request/response validation in the Sentiment Analysis API.
"""
from typing import Optional
from pydantic import BaseModel, Field, validator


class PredictRequest(BaseModel):
    """Request schema for sentiment prediction."""
    review: str = Field(..., min_length=1, description="Text to analyze for sentiment")
    language: str = Field(default="en", description="Language code (e.g., 'en', 'pt')")
    
    @validator('review')
    def review_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Review cannot be empty or whitespace only')
        return v
    
    @validator('language')
    def language_lowercase(cls, v):
        return v.lower().strip()


class PredictResponse(BaseModel):
    """Response schema for sentiment prediction."""
    prediction_id: str = Field(..., description="Unique identifier for this prediction")
    sentiment: str = Field(..., description="Predicted sentiment (POSITIVE/NEGATIVE)")
    probability: float = Field(..., ge=0.0, le=1.0, description="Confidence probability")
    translated_text: Optional[str] = Field(None, description="Translated text (if translation was performed)")
    original_text: str = Field(..., description="Original input text")


class FeedbackRequest(BaseModel):
    """Request schema for submitting feedback on predictions."""
    prediction_id: str = Field(..., description="UUID of the prediction to provide feedback for")
    correct_sentiment: int = Field(..., ge=0, le=1, description="Correct sentiment label (0=Negative, 1=Positive)")
    
    @validator('correct_sentiment')
    def validate_sentiment(cls, v):
        if v not in [0, 1]:
            raise ValueError('correct_sentiment must be 0 (Negative) or 1 (Positive)')
        return v


class FeedbackResponse(BaseModel):
    """Response schema for feedback submission."""
    status: str = Field(..., description="Status of feedback submission")
    message: str = Field(..., description="Detailed message")
    prediction_id: str = Field(..., description="UUID of the updated prediction")


class MetricsResponse(BaseModel):
    """Response schema for real-time metrics."""
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Accuracy score")
    f1_score: float = Field(..., ge=0.0, le=1.0, description="F1 score")
    total_predictions: int = Field(..., ge=0, description="Total number of predictions made")
    labeled_predictions: int = Field(..., ge=0, description="Number of predictions with feedback")
