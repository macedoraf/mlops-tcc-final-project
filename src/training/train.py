import os
import argparse
import pandas as pd
from feast import FeatureStore
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
import mlflow
import mlflow.sklearn

def train():
    # 1. Setup Paths and Config
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_store_path", type=str, default="/feature_store")
    parser.add_argument("--data_path", type=str, default="/data/processed/sentiment_features.parquet")
    parser.add_argument("--model_name", type=str, default="AmazonSentimentModel")
    args = parser.parse_args()

    print("Starting training pipeline...")
    print(f"Feature Store Path: {args.feature_store_path}")
    print(f"Data Path: {args.data_path}")

    # 2. Load Entity DataFrame
    # We need review_id and event_timestamp to fetch historical features
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found at {args.data_path}")
    
    entity_df = pd.read_parquet(args.data_path)
    print(f"Loaded entity dataframe with {len(entity_df)} records.")

    # Ensure timestamps are correct type
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"])

    # 3. Fetch Features from Feast
    store = FeatureStore(repo_path=args.feature_store_path)
    
    feature_vector = [
        "sentiment_features_view:text",
        "sentiment_features_view:polarity"
    ]

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=feature_vector
    ).to_df()

    print("Fetched historical features from Feast.")
    print(training_df.head())

    # Drop rows with missing values if any
    training_df.dropna(inplace=True)

    # 4. Prepare Data for Training
    X = training_df["text"]
    y = training_df["polarity"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Train Model
    # Use a Pipeline with TF-IDF and Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000)),
        ('clf', LogisticRegression())
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = pipeline.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted') # Use weighted for multi-class or binary safety
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Evaluation Metrics: F1={f1:.4f}, Accuracy={accuracy:.4f}")

    # 7. Log to MLflow
    # Set tracking URI to the service name in docker-compose network
    mlflow.set_tracking_uri("http://mlflow-server:5001")
    mlflow.set_experiment("sentiment_analysis_training")

    with mlflow.start_run():
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("vectorizer", "TfidfVectorizer")
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(pipeline, "model")

        # 8. Conditional Registration
        if f1 > 0.75:
            print(f"F1 Score ({f1:.4f}) > 0.75. Registering model as Production...")
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            mv = mlflow.register_model(model_uri, args.model_name)
            
            # Transition to Production
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=args.model_name,
                version=mv.version,
                stage="Production"
            )
            print(f"Model {args.model_name} version {mv.version} transitioned to Production.")
        else:
            print(f"F1 Score ({f1:.4f}) did not meet threshold (0.75). Skipping registration.")

if __name__ == "__main__":
    train()
