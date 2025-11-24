from datetime import datetime, timedelta
import pandas as pd
import os
import zipfile
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.datasets import Dataset
import logging

# Constants
ROOT_PATH = "/opt/airflow/data"
DATA_ZIP_PATH = f"{ROOT_PATH}/amazon-reviews.zip"
DATA_EXTRACTED_PATH = f"{ROOT_PATH}/extracted"
FEATURE_STORE_PATH = "/opt/airflow/feature_store"
PROCESSED_PATH = f"{ROOT_PATH}/processed/sentiment_features.parquet"
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/kritanjalijain/amazon-reviews"
# Ajuste o total de amostras por classe conforme necessário
SAMPLES_PER_CLASS = 25 * 1000 

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

# Configure logging
logger = logging.getLogger(__name__)

def process_data_logic(extracted_path, processed_path, samples_per_class=SAMPLES_PER_CLASS):
    """
    Core logic for processing data: reads CSVs, balances ONLY by polarity, 
    adds timestamps, and saves to Parquet.
    """
    try:
        logger.info(f"Starting balanced processing (Polarity Only). Target: {samples_per_class} per class.")
        
        # Dicionário simplificado para armazenar os textos por polaridade
        # 1: Negativo, 2: Positivo
        data_store = {
            1: [], 
            2: []
        }
        
        chunk_size = 100000
        chunks_processed = 0
        
        input_files = [f"{extracted_path}/train.csv", f"{extracted_path}/test.csv"]
        
        for file_path in input_files:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}. Skipping.")
                continue
                
            logger.info(f"Processing file: {file_path}")
            
            # Lê o arquivo em chunks para evitar estouro de memória
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, header=None, names=['polarity', 'title', 'text']):    
                chunks_processed += 1
                if chunks_processed % 10 == 0:
                    logger.info(f"Processing chunk {chunks_processed}...")
                    logger.info(f"Current counts -> Neg(1): {len(data_store[1])}, Pos(2): {len(data_store[2])}")

                # 1. Limpeza Básica
                chunk = chunk.dropna(subset=["text", "polarity"])
                chunk['text'] = chunk['text'].astype(str)
                chunk['title'] = chunk['title'].fillna("").astype(str)
                chunk['polarity'] = chunk['polarity'].astype(int)
                
                # 2. Enriquecer texto (Título + Texto)
                chunk['text'] = chunk['title'] + " " + chunk['text']
                
                # 3. Coleta balanceada por classe
                for sentiment in [1, 2]:
                    # Se já atingimos a cota para este sentimento, pule
                    if len(data_store[sentiment]) >= samples_per_class:
                        continue

                    # Filtra o chunk atual pelo sentimento
                    current_samples = chunk[chunk['polarity'] == sentiment]['text'].tolist()
                    
                    # Calcula quantos faltam para completar a cota
                    needed = samples_per_class - len(data_store[sentiment])
                    
                    # Adiciona o necessário (ou tudo o que tiver no chunk se for menos que o necessário)
                    if current_samples:
                        data_store[sentiment].extend(current_samples[:needed])

                # Critério de Parada: Se ambas as classes atingiram a cota
                if len(data_store[1]) >= samples_per_class and len(data_store[2]) >= samples_per_class:
                    logger.info(f"All classes balanced and filled at chunk {chunks_processed}!")
                    break
            
            # Quebra o loop de arquivos se já terminou
            if len(data_store[1]) >= samples_per_class and len(data_store[2]) >= samples_per_class:
                break
        
        # 4. Consolidação
        final_rows = []
        # Classe 1 (Original) -> 0 (Dataset Final)
        for text in data_store[1]:
            final_rows.append([text, 0])
            
        # Classe 2 (Original) -> 1 (Dataset Final)
        for text in data_store[2]:
            final_rows.append([text, 1])
        
        df_result = pd.DataFrame(final_rows, columns=['text', 'polarity'])
        
        # Embaralha final
        df_result = df_result.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Data processing complete. Final shape: {df_result.shape}")
        logger.info(f"Class distribution: \n{df_result['polarity'].value_counts()}")
        
        if df_result.empty:
            raise ValueError("No data processed. Check input files and sampling criteria.")

        # Add timestamps for Feast
        now = datetime.utcnow()
        df_result['event_timestamp'] = now
        df_result['created_timestamp'] = now
        
        # Add a unique ID if not present (using index as review_id)
        df_result['review_id'] = df_result.index.astype(str)
        
        # Mock feature: tfidf_sum (placeholder for actual feature engineering)
        df_result['tfidf_sum'] = df_result['text'].apply(lambda x: len(str(x).split()) * 0.1) 
        
        # Select columns matching definitions.py
        final_df = df_result[['review_id', 'text', 'polarity', 'tfidf_sum', 'event_timestamp', 'created_timestamp']]
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        
        logger.info(f"Saving processed data to {processed_path}...")
        final_df.to_parquet(processed_path)
        logger.info("Data saved successfully.")
        
    except Exception as e:
        logger.error(f"Error during data processing: {e}", exc_info=True)
        raise

def process_data_func():
    """
    Wrapper function for Airflow PythonOperator.
    """
    logger.info("Starting process_data_func task.")
    logger.info(f"Unzipping {DATA_ZIP_PATH} to {DATA_EXTRACTED_PATH}...")
    try:
        with zipfile.ZipFile(DATA_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_EXTRACTED_PATH)
    except Exception as e:
        logger.error(f"Unzip failed: {e}")
        raise
    
    process_data_logic(DATA_EXTRACTED_PATH, PROCESSED_PATH)
    logger.info("process_data_func task completed.")

with DAG(
    'ingest_to_feast',
    default_args=default_args,
    description='Ingest Amazon reviews to Feast Feature Store',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    # Task 1: Extract Data
    extract_data = BashOperator(
        task_id='extract_data',
        bash_command=f'curl -L -o {DATA_ZIP_PATH} {KAGGLE_URL}',
    )

    # Task 2: Process Data
    process_data = PythonOperator(
        task_id='process_data',
        python_callable=process_data_func,
    )

    # Task 2: Feast Apply (Register features)
    feast_apply = BashOperator(
        task_id='feast_apply',
        bash_command=f'cd {FEATURE_STORE_PATH} && /home/airflow/.local/bin/feast apply',
    )

    # Task 3: Feast Materialize (Load to Redis)
    # Materialize from 1 year ago to now to ensure all data is loaded
    feast_materialize = BashOperator(
        task_id='feast_materialize',
        bash_command=f'cd {FEATURE_STORE_PATH} && /home/airflow/.local/bin/feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")',
        outlets=[Dataset(f"file://{PROCESSED_PATH}")],
    )

    extract_data >> process_data >> feast_apply >> feast_materialize