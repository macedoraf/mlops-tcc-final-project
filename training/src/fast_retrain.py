import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from feast import FeatureStore
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

def get_production_model_params(model_name):
    """
    Fetches parameters from the current Production model in MLflow.
    """
    client = mlflow.tracking.MlflowClient()
    try:
        # Get the latest version in Production
        latest_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not latest_versions:
            print(f"No Production model found for {model_name}.")
            return None, None
        
        prod_version = latest_versions[0]
        run_id = prod_version.run_id
        
        # Get run info to extract params and metrics
        run = client.get_run(run_id)
        params = run.data.params
        metrics = run.data.metrics
        
        print(f"Found Production model version {prod_version.version} (Run ID: {run_id})")
        return params, metrics.get("final_test_f1", 0.0)
        
    except Exception as e:
        print(f"Error fetching production model: {e}")
        return None, None

def cast_params(params):
    """
    Casts MLflow string params to appropriate types for LogisticRegression and TfidfVectorizer.
    """
    casted = {}
    
    # LogisticRegression params
    if "C" in params:
        casted["C"] = float(params["C"])
    if "max_iter" in params:
        casted["max_iter"] = int(params["max_iter"])
    if "random_state" in params:
        casted["random_state"] = int(params["random_state"])
    if "solver" in params:
        casted["solver"] = params["solver"]
        
    # TfidfVectorizer params
    if "tfidf_max_features" in params:
        casted["tfidf_max_features"] = int(params["tfidf_max_features"])
    if "tfidf_ngram_max" in params:
        casted["tfidf_ngram_max"] = int(params["tfidf_ngram_max"])
        
    return casted

def fetch_data(feature_store_path, days=90):
    """
    Fetches data from Feast for the last `days` days.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    print(f"Fetching data from {start_date} to {end_date}...")
    
    try:
        # Check if feature store repo exists
        if os.path.exists(os.path.join(feature_store_path, "feature_store.yaml")):
            store = FeatureStore(repo_path=feature_store_path)
            # In a real scenario, use store.get_historical_features()
            # For now, we fall back to reading the parquet directly as we don't have entity source ready
            print("Feast repo found, but using direct parquet read for simplicity in this demo.")
        
        # Fallback / Direct read
        data_path = "/data/processed/sentiment_features.parquet"
        if os.path.exists(data_path):
            df = pd.read_parquet(data_path)
            if "event_timestamp" in df.columns:
                df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
                mask = (df["event_timestamp"] >= start_date) & (df["event_timestamp"] <= end_date)
                df_filtered = df.loc[mask]
                print(f"Fetched {len(df_filtered)} rows from offline store (parquet).")
                return df_filtered
            else:
                print("Warning: 'event_timestamp' not found in data. Using all data.")
                return df
        else:
            print(f"Data file not found at {data_path}.")
            return pd.DataFrame()

    except Exception as e:
        print(f"Error fetching data: {e}")
        # Last ditch effort: try to read the file anyway
        data_path = "/data/processed/sentiment_features.parquet"
        if os.path.exists(data_path):
             print("Attempting direct read after error...")
             return pd.read_parquet(data_path)
        return pd.DataFrame()

def train_and_evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_store_path", type=str, default="/feature_store")
    parser.add_argument("--model_name", type=str, default="AmazonSentimentModel_Best")
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5001"))
    mlflow.set_experiment("Sentiment_Analysis_Fast_Retrain")

    # 1. Fetch Config
    prod_params, prod_f1 = get_production_model_params(args.model_name)
    if not prod_params:
        print("Could not fetch production params. Aborting fast retrain.")
        return

    casted_params = cast_params(prod_params)
    print(f"Retraining with params: {casted_params}")

    # 2. Fetch Data
    df = fetch_data(args.feature_store_path)
    if df.empty:
        print("No data found. Aborting.")
        return

    df.dropna(inplace=True)
    X = df["text"]
    y = df["polarity"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # 3. Train
    # Reconstruct pipeline
    vect_params = {
        "max_features": casted_params.get("tfidf_max_features", 1000),
        "ngram_range": (1, casted_params.get("tfidf_ngram_max", 1))
    }
    
    clf_params = {k: v for k, v in casted_params.items() if k not in ["tfidf_max_features", "tfidf_ngram_max"]}
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(**vect_params)),
        ('clf', LogisticRegression(**clf_params))
    ])

    with mlflow.start_run(run_name="Fast_Retrain") as run:
        print("Training model...")
        pipeline.fit(X_train, y_train)

        # 4. Evaluate
        y_pred = pipeline.predict(X_test)
        new_f1 = f1_score(y_test, y_pred, average='weighted')
        new_acc = accuracy_score(y_test, y_pred)
        
        print(f"New F1: {new_f1:.4f} (Production F1: {prod_f1:.4f})")

        # Log metrics and params
        mlflow.log_params(casted_params)
        mlflow.log_metric("final_test_f1", new_f1)
        mlflow.log_metric("final_test_accuracy", new_acc)
        mlflow.set_tag("hyperparameters_source", "production_clone")
        mlflow.set_tag("training_mode", "fast_retrain")

        # 5. Promotion Logic
        # Allow slight drop (0.02) for freshness, or if absolute performance is high (>0.85)
        if new_f1 >= (prod_f1 - 0.02) or new_f1 > 0.85:
            print("Performance criteria met. Promoting model...")
            
            signature = infer_signature(X_test.to_frame(), y_pred)
            input_example = X_train.iloc[:5].to_frame()
            
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="best_model_artifact",
                signature=signature,
                input_example=input_example
            )
            
            model_uri = f"runs:/{run.info.run_id}/best_model_artifact"
            mv = mlflow.register_model(model_uri, args.model_name)
            print(f"Registered {args.model_name} version {mv.version}")
            
            # Optional: Transition to Production automatically? 
            # The prompt says "Register model version", usually implies available for use.
            # We can leave manual transition or automate it. 
            # For safety in this script, we just register.
        else:
            print("Performance degraded significantly. Model not promoted.")

if __name__ == "__main__":
    train_and_evaluate()
