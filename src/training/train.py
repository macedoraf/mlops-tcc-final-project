import os
import argparse
import pandas as pd
from feast import FeatureStore
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import re
import optuna

def train():
    # 1. Setup Paths and Config
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_store_path", type=str, default="/feature_store")
    parser.add_argument("--data_path", type=str, default="/data/processed/sentiment_features.parquet")
    parser.add_argument("--model_name", type=str, default="AmazonSentimentModel")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials")
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

    # 5. Hyperparameter Tuning with Optuna
    print(f"\nStarting Optuna hyperparameter optimization ({args.n_trials} trials)...")
    
    def objective(trial):
        """Optuna objective function for hyperparameter tuning."""
        # Suggest hyperparameters
        max_features = trial.suggest_int("max_features", 1000, 10000, step=1000)
        ngram_min = trial.suggest_int("ngram_min", 1, 2)
        ngram_max = trial.suggest_int("ngram_max", ngram_min, 3)
        min_df = trial.suggest_int("min_df", 1, 10)
        max_df = trial.suggest_float("max_df", 0.7, 0.95)
        
        C = trial.suggest_float("C", 0.1, 10.0, log=True)
        max_iter = trial.suggest_int("max_iter", 1000, 3000, step=500)
        
        # Create pipeline with suggested hyperparameters
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=max_features,
                ngram_range=(ngram_min, ngram_max),
                min_df=min_df,
                max_df=max_df
            )),
            ('clf', LinearSVC(
                C=C,
                max_iter=max_iter,
                random_state=42,
                dual="auto"
            ))
        ])
        
        # Cross-validation score
        scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='f1_weighted', n_jobs=-1)
        return scores.mean()
    
    # Run Optuna optimization
    study = optuna.create_study(direction="maximize", study_name="LinearSVC_tuning")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best F1 Score (CV): {study.best_trial.value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # 6. Train final model with best hyperparameters
    best_params = study.best_params
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=best_params["max_features"],
            ngram_range=(best_params["ngram_min"], best_params["ngram_max"]),
            min_df=best_params["min_df"],
            max_df=best_params["max_df"]
        )),
        ('clf', LinearSVC(
            C=best_params["C"],
            max_iter=best_params["max_iter"],
            random_state=42,
            dual="auto"
        ))
    ])

    print("\nTraining final model with optimized hyperparameters...")
    pipeline.fit(X_train, y_train)

    # 7. Evaluate
    y_pred = pipeline.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nFinal Evaluation Metrics: F1={f1:.4f}, Accuracy={accuracy:.4f}")

    # 8. Log to MLflow
    mlflow.set_tracking_uri("http://mlflow-server:5001")
    mlflow.set_experiment("sentiment_analysis_training")
    mlflow.sklearn.autolog()

    with mlflow.start_run() as run:
        # Log model type and hyperparameters
        mlflow.log_param("model_type", "LinearSVC")
        mlflow.log_param("vectorizer", "TfidfVectorizer")
        mlflow.log_param("optimization", "Optuna")
        mlflow.log_param("n_trials", args.n_trials)
        
        # Log best hyperparameters from Optuna
        for key, value in best_params.items():
            mlflow.log_param(f"best_{key}", value)
        
        # Log Optuna study metrics
        mlflow.log_metric("best_cv_f1_score", study.best_trial.value)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", accuracy)

        # Log model artifact
        mlflow.sklearn.log_model(pipeline, "model")
        
        # 9. Conditional Registration
        if f1 > 0.85:
            print(f"F1 Score ({f1:.4f}) > 0.85. Registering model as Production...")
            model_uri = f"runs:/{run.info.run_id}/model"
            mv = mlflow.register_model(model_uri, args.model_name)
            
            # Transition to Production
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=args.model_name,
                version=mv.version,
                stage="Production",
                archive_existing_versions=True
            )
            print(f"Model {args.model_name} version {mv.version} transitioned to Production.")
        else:
            print(f"F1 Score ({f1:.4f}) did not meet threshold (0.85). Skipping registration.")

if __name__ == "__main__":
    train()
