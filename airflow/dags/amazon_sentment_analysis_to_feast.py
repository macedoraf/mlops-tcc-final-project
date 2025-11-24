from datetime import datetime, timedelta
import pandas as pd
import os
import zipfile
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# Constants
ROOT_PATH = "/opt/airflow/data"
DATA_ZIP_PATH = f"{ROOT_PATH}/amazon-reviews.zip"
DATA_EXTRACTED_PATH = f"{ROOT_PATH}/extracted"
FEATURE_STORE_PATH = "/opt/airflow/feature_store"
PROCESSED_PATH = f"{ROOT_PATH}/processed/sentiment_features.parquet"
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/kritanjalijain/amazon-reviews"

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0    ,
    'retry_delay': timedelta(minutes=5),
}

import logging

# Configure logging
logger = logging.getLogger(__name__)

def process_data_logic(extracted_path, processed_path, samples_per_class=1000):
    """
    Core logic for processing data: reads CSVs, balances by class AND text length, adds timestamps, and saves to Parquet.
    Estratégia: 3 Bins (Curto, Médio, Longo).
    Meta: 1/3 das amostras para cada bin dentro de cada classe.
    """
    try:
        logger.info(f"Starting stratified data processing. Reading from {extracted_path}")
        
        # Definição dos Bins (Limites de caracteres)
        BIN_SHORT = 150
        BIN_MEDIUM = 600
        
        # Meta de amostras por sub-grupo (ex: Positivo-Curto, Positivo-Médio...)
        quota_per_subgroup = samples_per_class // 3
        
        # Dicionário para armazenar os dados separadamente
        # Estrutura: data_store[classe][bin]
        data_store = {
            1: {'short': [], 'medium': [], 'long': []}, # Classe 1 (Negativo no Raw)
            2: {'short': [], 'medium': [], 'long': []}  # Classe 2 (Positivo no Raw)
        }
        
        chunk_size = 100000
        chunks_processed = 0
        
        # We process both train and test files as a single source for simplicity in this pipeline, 
        # or we could process them separately. The original DAG combined them. 
        # Let's iterate over both files.
        input_files = [f"{extracted_path}/train.csv", f"{extracted_path}/test.csv"]
        
        for file_path in input_files:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}. Skipping.")
                continue
                
            logger.info(f"Processing file: {file_path}")
            
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, header=None, names=['polarity', 'title', 'text']):    
                chunks_processed += 1
                if chunks_processed % 10 == 0:
                    logger.info(f"Processing chunk {chunks_processed}...")

                # 1. Limpeza Básica
                chunk = chunk.dropna(subset=["text", "polarity"])
                chunk['text'] = chunk['text'].astype(str)
                chunk['title'] = chunk['title'].fillna("").astype(str)
                chunk['polarity'] = chunk['polarity'].astype(int)
                
                # 2. Calcular tamanho e enriquecer texto (Título + Texto)
                chunk['text'] = chunk['title'] + " " + chunk['text']
                chunk['char_len'] = chunk['text'].str.len()
                
                # 3. Iterar sobre as classes (1 e 2)
                for sentiment in [1, 2]:
                    # Filtra pelo sentimento atual
                    df_sent = chunk[chunk['polarity'] == sentiment]
                    
                    if df_sent.empty: continue

                    # --- Bin Curto ---
                    if len(data_store[sentiment]['short']) < quota_per_subgroup:
                        needed = quota_per_subgroup - len(data_store[sentiment]['short'])
                        subset = df_sent[df_sent['char_len'] < BIN_SHORT]
                        data_store[sentiment]['short'].extend(subset[['text', 'polarity']].head(needed).values.tolist())
                    
                    # --- Bin Médio ---
                    if len(data_store[sentiment]['medium']) < quota_per_subgroup:
                        needed = quota_per_subgroup - len(data_store[sentiment]['medium'])
                        subset = df_sent[(df_sent['char_len'] >= BIN_SHORT) & (df_sent['char_len'] < BIN_MEDIUM)]
                        data_store[sentiment]['medium'].extend(subset[['text', 'polarity']].head(needed).values.tolist())
                        
                    # --- Bin Longo ---
                    if len(data_store[sentiment]['long']) < quota_per_subgroup:
                        needed = quota_per_subgroup - len(data_store[sentiment]['long'])
                        subset = df_sent[df_sent['char_len'] >= BIN_MEDIUM]
                        data_store[sentiment]['long'].extend(subset[['text', 'polarity']].head(needed).values.tolist())

                # Critério de Parada: Verificamos se TODOS os baldes estão cheios
                full_buckets = 0
                total_buckets = 6 # 2 classes * 3 bins
                for s in [1, 2]:
                    for b in ['short', 'medium', 'long']:
                        if len(data_store[s][b]) >= quota_per_subgroup:
                            full_buckets += 1
                
                if full_buckets == total_buckets:
                    logger.info(f"All bins filled at chunk {chunks_processed}!")
                    break
            
            if full_buckets == total_buckets:
                break
        
        # 4. Consolidação
        final_rows = []
        for s in [1, 2]:
            # Mapeia label: 1(Neg) -> 0, 2(Pos) -> 1
            mapped_label = 0 if s == 1 else 1
            for b in ['short', 'medium', 'long']:
                # Adiciona ao resultado final ajustando o label
                for text, _ in data_store[s][b]:
                    final_rows.append([text, mapped_label])
        
        df_result = pd.DataFrame(final_rows, columns=['text', 'polarity'])
        
        # Embaralha final
        df_result = df_result.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Stratified sampling complete. Final shape: {df_result.shape}")
        
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
    )

    extract_data >> process_data >> feast_apply >> feast_materialize
