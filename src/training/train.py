import os
import argparse
import pandas as pd
import numpy as np
from feast import FeatureStore
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import optuna

def get_model_search_space(trial, model_name):
    """
    Define o espaço de busca do Optuna baseado no nome do modelo.
    Retorna um dicionário de kwargs para instanciar o classificador.
    """
    if model_name == "LinearSVC":
        return {
            "C": trial.suggest_float("C", 0.1, 10.0, log=True),
            "max_iter": trial.suggest_int("max_iter", 1000, 3000, step=500),
            "dual": "auto",
            "random_state": 42
        }
    elif model_name == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "random_state": 42
        }
    elif model_name == "LogisticRegression":
        return {
            "C": trial.suggest_float("C", 0.1, 10.0, log=True),
            "solver": "liblinear", # Good for text/sparse data
            "max_iter": 1000,
            "random_state": 42
        }
    elif model_name == "MultinomialNB":
        return {
            "alpha": trial.suggest_float("alpha", 0.01, 1.0)
        }
    return {}

def get_class_map():
    """Mapeia strings para classes do Scikit-Learn"""
    return {
        "LinearSVC": LinearSVC,
        "RandomForest": RandomForestClassifier,
        "LogisticRegression": LogisticRegression,
        "MultinomialNB": MultinomialNB
    }

def train():
    # 1. Setup Paths and Config
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_store_path", type=str, default="/feature_store")
    parser.add_argument("--data_path", type=str, default="/data/processed/sentiment_features.parquet")
    parser.add_argument("--model_name", type=str, default="AmazonSentimentModel_Best")
    parser.add_argument("--n_trials", type=int, default=5, help="Number of Optuna trials per model")
    args = parser.parse_args()

    # MLflow Setup
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5001"))
    mlflow.set_experiment("Sentiment_Analysis_AutoML")

    # 2. Load Data (Simulation or Feast)
    print("Loading data...")
    if not os.path.exists(args.data_path):
        # Mocking data creation if file doesn't exist for runnability
        print("Data file not found, creating mock data...")
        df_mock = pd.DataFrame({
            "text": ["I love this product", "Terrible, hate it", "It is okay", "Best purchase ever", "Waste of money"] * 20,
            "polarity": [1, 0, 1, 1, 0] * 20,
            "event_timestamp": pd.date_range(start="2023-01-01", periods=100)
        })
        df_mock.to_parquet(args.data_path)
        entity_df = df_mock
    else:
        entity_df = pd.read_parquet(args.data_path)

    # 3. Fetch Features from Feast (Optional: commented out if just testing logic without Feast repo)
    try:
        store = FeatureStore(repo_path=args.feature_store_path)
        feature_vector = ["sentiment_features_view:text", "sentiment_features_view:polarity"]
        # Assuming entity_df has keys. using entity_df directly for simplicity in this example context
        # training_df = store.get_historical_features(entity_df=entity_df, features=feature_vector).to_df()
        training_df = entity_df # Direct usage for simplicity if Feast not running
    except Exception as e:
        print(f"Feast skipped ({e}), using raw dataframe.")
        training_df = entity_df

    training_df.dropna(inplace=True)
    X = training_df["text"]
    y = training_df["polarity"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Models definition
    models_to_test = ["LinearSVC", "LogisticRegression", "RandomForest", "MultinomialNB"]
    class_map = get_class_map()

    best_overall_f1 = -1
    best_overall_pipeline = None
    best_overall_model_name = ""
    best_overall_params = {}

    # Start Parent Run (Orchestrator)
    with mlflow.start_run(run_name="Model_Selection_Workflow") as parent_run:
        print(f"Started Parent Run ID: {parent_run.info.run_id}")
        
        # Log Dataset info
        mlflow.log_param("dataset_path", args.data_path)
        mlflow.log_param("n_samples", len(training_df))
        
        # Log Input Example for Schema
        input_example = X_train.iloc[:5].to_frame()

        for model_name in models_to_test:
            print(f"\n>>> Optimizing {model_name}...")
            
            # Start Child Run for this specific model type
            with mlflow.start_run(run_name=f"Tuning_{model_name}", nested=True) as child_run:
                
                def objective(trial):
                    # 1. Vectorizer Params (Global tuning or Per model)
                    max_features = trial.suggest_int("tfidf_max_features", 1000, 5000, step=1000)
                    ngram_max = trial.suggest_int("tfidf_ngram_max", 1, 2)
                    
                    # 2. Model Params
                    clf_params = get_model_search_space(trial, model_name)
                    clf_class = class_map[model_name]

                    pipeline = Pipeline([
                        ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=(1, ngram_max))),
                        ('clf', clf_class(**clf_params))
                    ])

                    # Cross Validation
                    # Use n_jobs=1 to avoid memory exhaustion
                    scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='f1_weighted', n_jobs=1)
                    return scores.mean()

                # Run Optimization
                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=args.n_trials)

                # Log best results for this model type
                best_trial = study.best_trial
                print(f"   Best F1 for {model_name}: {best_trial.value:.4f}")
                
                mlflow.log_params(best_trial.params)
                mlflow.log_metric("best_cv_f1", best_trial.value)
                mlflow.log_param("model_type", model_name)

                # Check if this is the global winner
                if best_trial.value > best_overall_f1:
                    best_overall_f1 = best_trial.value
                    best_overall_model_name = model_name
                    best_overall_params = best_trial.params

        print("\n" + "="*50)
        print(f"WINNER: {best_overall_model_name} with F1: {best_overall_f1:.4f}")
        print("="*50)

        # ---------------------------------------------------------
        # Retrain the Winner on Full Training Data & Log Artifact
        # ---------------------------------------------------------
        print("Retraining best model for final artifact logging...")
        
        # Reconstruct pipeline
        # Separate TFIDF params from Model params based on naming convention used in objective
        tfidf_params = {k: v for k, v in best_overall_params.items() if k.startswith("tfidf")}
        model_params = {k: v for k, v in best_overall_params.items() if not k.startswith("tfidf")}
        
        # Clean up param names for instantiation
        final_vect_params = {
            "max_features": tfidf_params["tfidf_max_features"],
            "ngram_range": (1, tfidf_params["tfidf_ngram_max"])
        }
        
        # Need to clean keys for model (remove prefix if any, but our objective kept them clean)
        final_model_class = class_map[best_overall_model_name]
        
        final_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(**final_vect_params)),
            ('clf', final_model_class(**model_params))
        ])
        
        final_pipeline.fit(X_train, y_train)
        
        # Final Evaluation
        y_pred = final_pipeline.predict(X_test)
        final_f1 = f1_score(y_test, y_pred, average='weighted')
        final_acc = accuracy_score(y_test, y_pred)
        
        print(f"Final Test Set Metrics - F1: {final_f1:.4f}, Acc: {final_acc:.4f}")

        # Infer Signature (Schema)
        # We pass the input example (pandas DataFrame or Series) and the prediction
        signature = infer_signature(X_test.to_frame(), y_pred)

        # Log Final "Best Model" specific tags/metrics to Parent Run
        mlflow.log_metric("final_test_f1", final_f1)
        mlflow.log_metric("final_test_accuracy", final_acc)
        mlflow.log_param("winner_model", best_overall_model_name)
        
        # Log the Model with Signature and Input Example
        mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            artifact_path="best_model_artifact",
            signature=signature,
            input_example=input_example
        )

        # Register if good enough
        if final_f1 > 0.85:
            print("Met threshold. Registering model version...")
            model_uri = f"runs:/{parent_run.info.run_id}/best_model_artifact"
            mv = mlflow.register_model(model_uri, args.model_name)
            print(f"Registered {args.model_name} version {mv.version}")

if __name__ == "__main__":
    train()