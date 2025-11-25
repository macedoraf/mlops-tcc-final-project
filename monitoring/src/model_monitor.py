import os
import time
import json
import logging
import pandas as pd
from typing import List, Dict, Any
from prometheus_client import start_http_server, Gauge
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("model_monitor")

# Configuration
LOGS_FILE = os.getenv("PRODUCTION_LOGS_FILE", "/app/logs/production_logs.jsonl")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "60"))
METRICS_PORT = int(os.getenv("METRICS_PORT", "8001"))

# Prometheus Metrics
MODEL_ACCURACY = Gauge('model_accuracy', 'Model accuracy based on feedback')
MODEL_F1_SCORE = Gauge('model_f1_score', 'Model F1 score based on feedback')
MODEL_PRECISION = Gauge('model_precision', 'Model precision based on feedback')
MODEL_RECALL = Gauge('model_recall', 'Model recall based on feedback')
MODEL_PREDICTIONS_TOTAL = Gauge('model_predictions_total', 'Total number of predictions logged')
MODEL_PREDICTIONS_LABELED = Gauge('model_predictions_labeled', 'Number of predictions with feedback')

# Confusion Matrix Metrics
CM_TP = Gauge('model_confusion_matrix_tp', 'True Positives')
CM_FP = Gauge('model_confusion_matrix_fp', 'False Positives')
CM_TN = Gauge('model_confusion_matrix_tn', 'True Negatives')
CM_FN = Gauge('model_confusion_matrix_fn', 'False Negatives')

def load_logs() -> pd.DataFrame:
    """Load logs from JSONL file."""
    if not os.path.exists(LOGS_FILE):
        logger.warning(f"Logs file not found at {LOGS_FILE}")
        return pd.DataFrame()
    
    entries = []
    try:
        with open(LOGS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        logger.error(f"Error reading logs file: {e}")
        return pd.DataFrame()
        
    return pd.DataFrame(entries)

def calculate_and_update_metrics():
    """Calculate metrics from logs and update Prometheus gauges."""
    logger.info("Starting metrics calculation cycle...")
    df = load_logs()
    
    if df.empty:
        logger.info("No logs found.")
        return

    # Total predictions
    total_preds = len(df)
    MODEL_PREDICTIONS_TOTAL.set(total_preds)
    
    # Filter for labeled data
    if 'actual_label' not in df.columns:
        logger.info("No 'actual_label' column in logs.")
        return

    labeled_df = df[df['actual_label'].notna()].copy()
    labeled_count = len(labeled_df)
    MODEL_PREDICTIONS_LABELED.set(labeled_count)
    
    if labeled_count == 0:
        logger.info("No labeled predictions found.")
        return

    try:
        y_true = labeled_df['actual_label'].astype(int)
        y_pred = labeled_df['prediction'].astype(int)
        
        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        
        # Update Gauges
        MODEL_ACCURACY.set(acc)
        MODEL_F1_SCORE.set(f1)
        MODEL_PRECISION.set(prec)
        MODEL_RECALL.set(rec)
        
        # Confusion Matrix
        # Labels: 0 (Negative), 1 (Positive)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        CM_TP.set(tp)
        CM_FP.set(fp)
        CM_TN.set(tn)
        CM_FN.set(fn)
        
        logger.info(f"Metrics updated: Acc={acc:.4f}, F1={f1:.4f}, Labeled={labeled_count}")
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")

def main():
    """Main loop."""
    logger.info(f"Starting Model Monitor on port {METRICS_PORT}")
    start_http_server(METRICS_PORT)
    
    while True:
        try:
            calculate_and_update_metrics()
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
        
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
