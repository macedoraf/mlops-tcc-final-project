"""
FastAPI application for Sentiment Analysis model serving.
Includes translation layer for multi-language support and feedback loop for monitoring.
"""
import os
import pickle
import logging
from typing import Optional, Tuple, Any
from uuid import uuid4

import mlflow.sklearn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from deep_translator import GoogleTranslator
from prometheus_fastapi_instrumentator import Instrumentator

from src.serving.schemas import (
    PredictRequest,
    PredictResponse,
    FeedbackRequest,
    FeedbackResponse,
    MetricsResponse
)
from src.serving.monitor import log_prediction, update_feedback, calculate_metrics

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app


app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment analysis with MLflow integration",
    version="1.0.0"
)

# Instrumentator for Prometheus
Instrumentator().instrument(app).expose(app)

# Global model variables
model = None
model_version = ""

# Translation cache to avoid repeated translations
translation_cache = {}


def load_model_from_mlflow() -> Tuple[Optional[Any], Optional[str]]:
    """
    Attempt to load model from MLflow registry.
    Uses models:/ModelName/Version format.
    
    Returns:
        Tuple of (loaded model, version) or (None, None) if failed
    """
    try:
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5001")
        mlflow.set_tracking_uri(mlflow_uri)
        
        model_name = os.getenv("MLFLOW_MODEL_NAME", "SentimentAnalysis_Production")
        model_version = "5"
        
        logger.info(f"Attempting to load model {model_name} version {model_version} from MLflow at {mlflow_uri}")
        loaded_model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")
        logger.info(f"Successfully loaded model from MLflow: {model_name}/{model_version}")
        
        return loaded_model, model_version
        
    except Exception as e:
        logger.warning(f"Failed to load model from MLflow: {e}")
        return None, None

def load_model():
    """Load model from MLflow with fallback to local file."""
    global model, model_version
    
    # Try MLflow first
    model, model_version = load_model_from_mlflow()
    
    if model is None:
        logger.error("Failed to load model from both MLflow and local file!")
    else:
        logger.info(f"Model loaded successfully (version: {model_version})")


@app.on_event("startup")
async def startup_event():
    """Initialize model on application startup."""
    load_model()


def translate_text(text: str, source_lang: str, target_lang: str = "en") -> Optional[str]:
    """
    Translate text from source language to target language.
    
    Args:
        text: Text to translate
        source_lang: Source language code
        target_lang: Target language code (default: "en")
    
    Returns:
        Translated text or None if translation fails
    """
    # Check cache first
    cache_key = f"{source_lang}:{text}"
    if cache_key in translation_cache:
        logger.debug(f"Using cached translation for: {text[:50]}...")
        return translation_cache[cache_key]
    
    try:
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated = translator.translate(text)
        
        # Cache the translation
        translation_cache[cache_key] = translated
        
        logger.info(f"Translated text from {source_lang} to {target_lang}")
        return translated
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return None


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, background_tasks: BackgroundTasks):
    """
    Predict sentiment for given text with optional translation.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable."
        )

    prediction_id = str(uuid4())
    original_text = request.review
    text_to_predict = original_text

    try:
        if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
            raise Exception("Loaded model does not support prediction. Check MLflow saving.")

        logger.info(f"Predicting for text: {text_to_predict}")
        prediction_proba = model.predict_proba([text_to_predict])[0]
        prediction_class = int(model.predict([text_to_predict])[0])

        if prediction_class == 1:
            sentiment = "NEGATIVE"
            prediction_binary = 0
        elif prediction_class == 2:
            sentiment = "POSITIVE"
            prediction_binary = 1
        else:
            # fallback: assume >1 é positivo
            sentiment = "POSITIVE" if prediction_class > 1 else "NEGATIVE"
            prediction_binary = 1 if prediction_class > 1 else 0

        probability = float(max(prediction_proba))

        # 5. Log em background
        background_tasks.add_task(
            log_prediction,
            prediction_id=prediction_id,
            original_text=original_text,
            prediction=prediction_binary,
            probability=probability,
            model_version=model_version
        )

        # 6. Retorno estruturado
        return PredictResponse(
            prediction_id=prediction_id,
            sentiment=sentiment,
            probability=probability,
            original_text=original_text
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest):
    """
    Submit feedback for a prediction.
    
    Args:
        request: Feedback request with prediction_id and correct_sentiment
    
    Returns:
        Feedback response with status and message
    """
    success, message = update_feedback(request.prediction_id, request.correct_sentiment)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=message
        )
    
    return FeedbackResponse(
        status="success",
        message=message,
        prediction_id=request.prediction_id
    )


@app.get("/metrics/realtime", response_model=MetricsResponse)
async def get_realtime_metrics():
    """
    Get real-time performance metrics based on feedback.
    
    Returns:
        Metrics response with accuracy, F1 score, and prediction counts
    """
    try:
        metrics = calculate_metrics()
        
        return MetricsResponse(
            accuracy=metrics["accuracy"],
            f1_score=metrics["f1_score"],
            total_predictions=metrics["total_predictions"],
            labeled_predictions=metrics["labeled_predictions"]
        )
        
    except Exception as e:
        logger.error(f"Failed to calculate metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate metrics: {str(e)}"
        )


@app.get("/health")
async def health():
    """
    Health check endpoint.
    
    Returns:
        Health status and model information
    """
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": model_version
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Sentiment Analysis API",
        "version": "1.0.0",
        "description": "Multi-language sentiment analysis with feedback loop",
        "endpoints": {
            "predict": "/predict",
            "feedback": "/feedback",
            "metrics": "/metrics/realtime",
            "health": "/health",
            "docs": "/docs"
        }
    }
