import os
import pandas as pd
import mlflow
import mlflow.sklearn
from feast import FeatureStore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime

# Constants
FEATURE_STORE_PATH = "src/feature_store"
MLFLOW_TRACKING_URI = "http://localhost:5001"
EXPERIMENT_NAME = "Amazon_Sentiment_Analysis"

def train_model():
    # 1. Initialize Feast Feature Store
    print("Initializing Feature Store...")
    store = FeatureStore(repo_path=FEATURE_STORE_PATH)

    # 2. Define entity dataframe for training (Population)
    # In a real scenario, this would come from an upstream task or a list of IDs we want to train on.
    # For this demo, we'll read the processed parquet directly to get the IDs and timestamps,
    # effectively simulating "selecting all available data".
    # Note: In production, you shouldn't read the offline store directly like this usually, 
    # but we need a list of entity_ids and timestamps to query Feast.
    
    print("Loading entity dataframe...")
    # Assuming the Airflow DAG has run and created this file
    parquet_path = "data/processed/sentiment_features.parquet"
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"{parquet_path} not found. Run Airflow DAG first.")
    
    entities_df = pd.read_parquet(parquet_path)
    entities_df = entities_df[['review_id', 'event_timestamp']]

    # 3. Fetch historical features from Feast
    print("Fetching historical features from Feast...")
    training_df = store.get_historical_features(
        entity_df=entities_df,
        features=[
            "sentiment_features:text",
            "sentiment_features:polarity",
        ],
    ).to_df()

    print(f"Fetched {len(training_df)} rows.")
    
    # Drop rows with missing values if any
    training_df.dropna(inplace=True)

    # 4. Prepare Data
    X = training_df['text']
    y = training_df['polarity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Train Model (Pipeline)
    print("Training model...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000)),
        ('clf', LogisticRegression())
    ])

    pipeline.fit(X_train, y_train)

    # 6. Evaluate
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))

    # 7. Log to MLflow
    print(f"Logging to MLflow at {MLFLOW_TRACKING_URI}...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_param("source", "feast_feature_store")
        mlflow.log_metric("accuracy", accuracy)
        
        mlflow.sklearn.log_model(
            pipeline, 
            "model",
            registered_model_name="SentimentAnalysis_Feast"
        )
        print("Model logged successfully.")

if __name__ == "__main__":
    train_model()
