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
        
        model_name = "AmazonSentimentModel_Best"
        stage = "Production"
        
        logger.info(f"Attempting to load model {model_name} (Stage: {stage}) from MLflow at {mlflow_uri}")
        # Load the model from the Production stage
        loaded_model = mlflow.sklearn.load_model(f"models:/{model_name}/{stage}")
        
        # Get the actual version number for logging (optional, requires client)
        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions(model_name, stages=[stage])
        if latest_versions:
            model_version = latest_versions[0].version
        else:
            model_version = "unknown"
            
        logger.info(f"Successfully loaded model from MLflow: {model_name}/{stage} (Version: {model_version})")
        
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
        if not hasattr(model, "predict"):
            raise Exception("Loaded model does not support prediction. Check MLflow saving.")

        logger.info(f"Predicting for text: {text_to_predict}")
        prediction_class = int(model.predict([text_to_predict])[0])
        
        # Calculate confidence/probability
        if hasattr(model, "predict_proba"):
            # For models with predict_proba (e.g., LogisticRegression)
            prediction_proba = model.predict_proba([text_to_predict])[0]
            probability = float(max(prediction_proba))
        elif hasattr(model, "decision_function"):
            # For LinearSVC and similar models
            decision_values = model.decision_function([text_to_predict])[0]
            # Convert decision function to pseudo-probability using sigmoid
            if isinstance(decision_values, (list, tuple)) or hasattr(decision_values, '__len__'):
                # Multi-class case
                import numpy as np
                exp_values = np.exp(decision_values - np.max(decision_values))
                probabilities = exp_values / exp_values.sum()
                probability = float(max(probabilities))
            else:
                # Binary case
                from math import exp
                probability = 1 / (1 + exp(-abs(decision_values)))
        else:
            # Fallback
            probability = 0.5

        mapped_classes = {
            0: "NEGATIVE",
            1: "POSITIVE"
        }

        sentiment = mapped_classes.get(prediction_class, "UNKNOWN")
        prediction_binary = prediction_class

        probability = float(probability)

        # 5. Log em background
        background_tasks.add_task(
            log_prediction,
            prediction_id=prediction_id,
            original_text=original_text,
            prediction=prediction_binary,
            probability=probability,
            model_version=model_version,
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


@app.post("/reload")
async def reload_model():
    """
    Force reload of the model from MLflow (Production stage).
    Useful for continuous deployment without restarting the container.
    """
    global model, model_version
    try:
        new_model, new_version = load_model_from_mlflow()
        if new_model:
            model = new_model
            model_version = new_version
            return {"status": "success", "message": f"Model reloaded successfully. Version: {model_version}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to load model from MLflow")
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
            "reload": "/reload",
            "docs": "/docs"
        }
    }
