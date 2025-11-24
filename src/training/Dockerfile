FROM python:3.11-slim

# Install system dependencies if needed (e.g. for pyarrow or pandas build)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    "numpy<2" \
    "protobuf>=5.26.0" \
    pandas \
    scikit-learn \
    feast \
    mlflow \
    pyarrow

# Set working directory
WORKDIR /app

# No ENTRYPOINT, as Airflow will override it
