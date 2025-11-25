FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for scikit-learn and pandas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /app/logs /app/models

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Environment variables
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5001
ENV MLFLOW_MODEL_NAME=SentimentAnalysis_Production
ENV MLFLOW_MODEL_VERSION=2
ENV LOCAL_MODEL_PATH=/app/models/model_pipeline.pkl
ENV PRODUCTION_LOGS_FILE=/app/logs/production_logs.jsonl

# Expose port
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
