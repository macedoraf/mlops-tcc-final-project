from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

from airflow.datasets import Dataset

dag = DAG(
    'continuous_training_dag',
    default_args=default_args,
    description='A simple Continuous Training DAG',
    schedule=[Dataset("file:///opt/airflow/data/processed/sentiment_features.parquet")],
    catchup=False,
)

# Task 2: Run Training in Docker
# We need to mount the host paths to the container.
# Since we are running Docker-in-Docker (sibling), we use the host's absolute paths.
HOST_PROJECT_PATH = "/Users/rafael/projects/machine-learning-tcc"

train_model = DockerOperator(
    task_id='train_model',
    image='sentiment-analysis-training:latest',
    api_version='auto',
    auto_remove=True,
    command='python3 /app/train.py',
    docker_url='unix://var/run/docker.sock',
    network_mode='machine-learning-tcc_mlops-network', # Must match the network name in docker-compose
    mounts=[
        Mount(source=f'{HOST_PROJECT_PATH}/feature_store', target='/feature_store', type='bind', read_only=True),
        Mount(source=f'{HOST_PROJECT_PATH}/data', target='/data', type='bind', read_only=True),
        Mount(source=f'{HOST_PROJECT_PATH}/src/training', target='/app', type='bind', read_only=False),
        Mount(source=f'{HOST_PROJECT_PATH}/mlflow_data', target='/mlflow_data', type='bind', read_only=False),
    ],
    environment={
        'MLFLOW_TRACKING_URI': 'http://mlflow-server:5001'
    },
    dag=dag,
)

train_model
