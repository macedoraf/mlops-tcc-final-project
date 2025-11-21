# End-to-End MLOps Project: Sentiment Analysis

This project implements a complete MLOps pipeline for Sentiment Analysis using the Amazon Reviews dataset. It features a model serving API, a user-friendly frontend, and a comprehensive observability stack.

## Stack
- **Serving**: FastAPI (Model Inference)
- **Frontend**: Streamlit (User Interface)
- **Tracking**: MLflow (Model Registry & Experiment Tracking)
- **Monitoring**: Prometheus + Grafana (Infrastructure & Model Quality)
- **Infrastructure**: Docker Compose

## Directory Structure
- `src/`: Source code
  - `serving/`: FastAPI application
  - `frontend/`: Streamlit application
  - `monitoring/`: Model monitoring service
  - `utils/`: Utility scripts
- `notebooks/`: Jupyter notebooks for data exploration and training
- `monitoring/`: Prometheus and Grafana configurations
- `docker-compose.yml`: Service orchestration

## How to Run

1. **Start Infrastructure**:
   ```bash
   docker compose up --build -d
   ```

2. **Access Services**:
   - **Frontend**: [http://localhost:8501](http://localhost:8501) (Interact with the model here)
   - **API**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **MLflow**: [http://localhost:5001](http://localhost:5001)
   - **Grafana**: [http://localhost:3000](http://localhost:3000) (User/Pass: `admin`/`admin`)
   - **Prometheus**: [http://localhost:9090](http://localhost:9090)
   - **Model Monitor Metrics**: [http://localhost:8001](http://localhost:8001)

3. **Train the Model**:
   - The model training is handled in `notebooks/01_data_and_training.ipynb`.
   - You can run this notebook to train a new model and log it to MLflow.
   - Ensure the `MLFLOW_TRACKING_URI` is set to `http://localhost:5001`.

4. **Monitoring & Feedback**:
   - Use the Frontend to send predictions.
   - The application supports a "Feedback Loop" (simulated or manual).
   - **Infrastructure Metrics** (Latency, Throughput) are visible in Grafana/Prometheus.
   - **Model Metrics** (Accuracy, F1) are calculated by the `model-monitor` service based on feedback logs.

## Observability
- **Prometheus** scrapes:
  - API metrics at `http://api-serving:8000/metrics`
  - Model metrics at `http://model-monitor:8001`
- **Alerting**: Configured for High Latency, Error Rates, and Model Accuracy Degradation.
