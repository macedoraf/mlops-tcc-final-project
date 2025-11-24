from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from docker.types import Mount

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'continuous_training_dag',
    default_args=default_args,
    description='A simple Continuous Training DAG',
    schedule_interval='@daily',
    catchup=False,
)

# Task 1: Wait for processed data
# Note: FileSensor checks for file existence. 
# The path is inside the Airflow container.
# We mapped ./data to /opt/airflow/data in docker-compose.
wait_for_data = FileSensor(
    task_id='wait_for_data',
    filepath='/opt/airflow/data/processed/  .parquet',
    poke_interval=60,
    timeout=600,
    mode='poke',
    dag=dag,
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
    network_mode='mlops-network', # Must match the network name in docker-compose
    mounts=[
        Mount(source=f'{HOST_PROJECT_PATH}/feature_store', target='/feature_store', type='bind', read_only=True),
        Mount(source=f'{HOST_PROJECT_PATH}/data', target='/data', type='bind', read_only=True),
        Mount(source=f'{HOST_PROJECT_PATH}/src/training', target='/app', type='bind', read_only=False),
    ],
    environment={
        'MLFLOW_TRACKING_URI': 'http://mlflow-server:5001'
    },
    dag=dag,
)

wait_for_data >> train_model
