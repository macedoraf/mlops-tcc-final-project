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

dag = DAG(
    'fast_retrain_dag',
    default_args=default_args,
    description='Fast Retraining DAG using Production Hyperparameters',
    schedule_interval='@weekly',
    catchup=False,
)

HOST_PROJECT_PATH = "/Users/rafael/projects/machine-learning-tcc"

fast_retrain_task = DockerOperator(
    task_id='fast_retrain_model',
    image='sentiment-analysis-training:latest',
    api_version='auto',
    auto_remove=True,
    command='python3 /app/src/fast_retrain.py',
    docker_url='unix://var/run/docker.sock',
    network_mode='machine-learning-tcc_mlops-network',
    mounts=[
        Mount(source=f'{HOST_PROJECT_PATH}/feature_store', target='/feature_store', type='bind', read_only=True),
        Mount(source=f'{HOST_PROJECT_PATH}/data', target='/data', type='bind', read_only=True),
        Mount(source=f'{HOST_PROJECT_PATH}/training', target='/app', type='bind', read_only=False),
        Mount(source=f'{HOST_PROJECT_PATH}/mlflow_data', target='/mlflow_data', type='bind', read_only=False),
    ],
    environment={
        'MLFLOW_TRACKING_URI': 'http://mlflow-server:5001'
    },
    mem_limit='2g', # Strict memory limit as requested
    dag=dag,
)

fast_retrain_task
