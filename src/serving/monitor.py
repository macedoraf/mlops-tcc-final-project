"""
Monitoring utilities for logging predictions and calculating performance metrics.
"""
import os
import json
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)

# Production logs file path
PRODUCTION_LOGS_FILE = os.getenv("PRODUCTION_LOGS_FILE", "/app/logs/production_logs.jsonl")


def ensure_logs_directory():
    """Ensure the logs directory exists."""
    os.makedirs(os.path.dirname(PRODUCTION_LOGS_FILE), exist_ok=True)


async def log_prediction(
    prediction_id: str,
    original_text: str,
    prediction: int,
    probability: float,
    model_version: str
) -> None:
    """
    Log prediction details to JSONL file asynchronously.
    
    Args:
        prediction_id: Unique identifier for the prediction
        original_text: Original input text
        prediction: Predicted class (0 or 1)
        probability: Prediction probability/confidence
        model_version: Version of the model used
    """
    try:
        ensure_logs_directory()
        
        log_entry = {
            "prediction_id": prediction_id,
            "timestamp": datetime.utcnow().isoformat(),
            "original_text": original_text,
            "prediction": int(prediction),
            "probability": float(probability),
            "model_version": model_version,
            "actual_label": None  # Will be updated via feedback
        }
        
        with open(PRODUCTION_LOGS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
        logger.info(f"Logged prediction {prediction_id}")
        
    except Exception as e:
        logger.error(f"Error logging prediction {prediction_id}: {e}")


def update_feedback(prediction_id: str, actual_label: int) -> Tuple[bool, str]:
    """
    Update a prediction entry with the actual label from feedback.
    
    Args:
        prediction_id: UUID of the prediction to update
        actual_label: Correct sentiment label (0 or 1)
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        ensure_logs_directory()
        
        if not os.path.exists(PRODUCTION_LOGS_FILE):
            return False, "No predictions logged yet"
        
        # Read all entries
        entries = []
        found = False
        
        with open(PRODUCTION_LOGS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry.get("prediction_id") == prediction_id:
                        entry["actual_label"] = actual_label
                        entry["feedback_timestamp"] = datetime.utcnow().isoformat()
                        found = True
                    entries.append(entry)
        
        if not found:
            return False, f"Prediction ID {prediction_id} not found"
        
        # Write back all entries
        with open(PRODUCTION_LOGS_FILE, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        logger.info(f"Updated feedback for prediction {prediction_id}")
        return True, "Feedback recorded successfully"
        
    except Exception as e:
        logger.error(f"Error updating feedback for {prediction_id}: {e}")
        return False, f"Error updating feedback: {str(e)}"


def calculate_metrics() -> Dict[str, Any]:
    """
    Calculate real-time performance metrics from logged predictions.
    
    Returns:
        Dictionary containing accuracy, f1_score, total_predictions, and labeled_predictions
    """
    try:
        ensure_logs_directory()
        
        if not os.path.exists(PRODUCTION_LOGS_FILE):
            return {
                "accuracy": 0.0,
                "f1_score": 0.0,
                "total_predictions": 0,
                "labeled_predictions": 0
            }
        
        # Read JSONL file into DataFrame
        entries = []
        with open(PRODUCTION_LOGS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        
        if not entries:
            return {
                "accuracy": 0.0,
                "f1_score": 0.0,
                "total_predictions": 0,
                "labeled_predictions": 0
            }
        
        df = pd.DataFrame(entries)
        total_predictions = len(df)
        
        # Filter entries with actual labels (feedback received)
        labeled_df = df[df['actual_label'].notna()]
        labeled_predictions = len(labeled_df)
        
        if labeled_predictions == 0:
            return {
                "accuracy": 0.0,
                "f1_score": 0.0,
                "total_predictions": total_predictions,
                "labeled_predictions": 0
            }
        
        # Calculate metrics
        y_true = labeled_df['actual_label'].astype(int).tolist()
        y_pred = labeled_df['prediction'].astype(int).tolist()
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0.0)
        
        logger.info(f"Calculated metrics: Accuracy={acc:.4f}, F1={f1:.4f}, Labeled={labeled_predictions}/{total_predictions}")
        
        return {
            "accuracy": float(acc),
            "f1_score": float(f1),
            "total_predictions": total_predictions,
            "labeled_predictions": labeled_predictions
        }
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {
            "accuracy": 0.0,
            "f1_score": 0.0,
            "total_predictions": 0,
            "labeled_predictions": 0,
            "error": str(e)
        }
