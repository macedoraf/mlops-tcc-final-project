import pandas as pd
import os
import pytest
from datetime import datetime
from airflow.models import DagBag
from ..main.amazon_sentment_analysis_to_feast import process_data_logic

def test_process_data_logic(tmp_path):
    """
    Test the process_data_logic function with stratified sampling.
    """
    extracted_path = tmp_path / "extracted"
    extracted_path.mkdir()
    processed_path = tmp_path / "processed" / "processed_data.parquet"
    
    # Create dummy CSV data with varying lengths to fit bins
    # Short < 150, Medium 150-600, Long >= 600
    
    short_text = "Short text " * 5 # ~55 chars
    medium_text = "Medium text " * 30 # ~360 chars
    long_text = "Long text " * 70 # ~700 chars
    
    data = []
    # Create enough data to fill quotas
    # We'll set samples_per_class = 6, so quota per bin = 2
    # We need at least 2 of each (class, bin) combo
    
    for _ in range(5):
        data.append([1, "Title", short_text]) # Neg, Short
        data.append([1, "Title", medium_text]) # Neg, Medium
        data.append([1, "Title", long_text]) # Neg, Long
        data.append([2, "Title", short_text]) # Pos, Short
        data.append([2, "Title", medium_text]) # Pos, Medium
        data.append([2, "Title", long_text]) # Pos, Long
        
    df = pd.DataFrame(data, columns=['polarity', 'title', 'text'])
    
    # Save as train.csv and test.csv (duplicate for simplicity)
    df.to_csv(extracted_path / "train.csv", index=False, header=False)
    df.to_csv(extracted_path / "test.csv", index=False, header=False)
    
    # Run the processing logic with small quota
    process_data_logic(str(extracted_path), str(processed_path), samples_per_class=6)
    
    # Verify output
    assert processed_path.exists()
    df_result = pd.read_parquet(processed_path)
    
    # Check columns
    expected_cols = ['review_id', 'review_text', 'sentiment_label', 'tfidf_sum', 'event_timestamp', 'created_timestamp']
    for col in expected_cols:
        assert col in df_result.columns
        
    # Check balance
    # Quota per bin = 6 // 3 = 2
    # Total expected = 2 (quota) * 3 (bins) * 2 (classes) = 12
    # However, our input might be split across train/test. 
    # The logic iterates train then test.
    # We put 5 of each type in train and 5 in test. Total 10 of each type available.
    # Quota is 2. So we should get exactly 12 rows.
    
    assert len(df_result) == 12
    
    # Check class balance
    assert df_result['sentiment_label'].value_counts()[0] == 6
    assert df_result['sentiment_label'].value_counts()[1] == 6
    
    # Check bin balance (approximate check by length)
    # We can't easily check bins exactly without re-calculating, but we can check we have variation
    lengths = df_result['review_text'].str.len()
    assert (lengths < 150).any()
    assert ((lengths >= 150) & (lengths < 600)).any()
    assert (lengths >= 600).any()

def test_dag_structure():
    """
    Test to verify if the DAG loads correctly and has the expected structure.
    """
    dag_bag = DagBag(dag_folder="dags/main", include_examples=False)
    dag = dag_bag.get_dag(dag_id="ingest_to_feast")
    
    assert dag is not None
    assert len(dag.tasks) == 5
    
    # Check task IDs
    task_ids = set(task.task_id for task in dag.tasks)
    expected_task_ids = {'install_dependencies', 'extract_data', 'process_data', 'feast_apply', 'feast_materialize'}
    assert task_ids == expected_task_ids
    
    # Check dependencies
    install_deps = dag.get_task('install_dependencies')
    extract_data = dag.get_task('extract_data')
    process_data = dag.get_task('process_data')
    feast_apply = dag.get_task('feast_apply')
    feast_materialize = dag.get_task('feast_materialize')
    
    assert extract_data in install_deps.downstream_list
    assert process_data in extract_data.downstream_list
    assert feast_apply in process_data.downstream_list
    assert feast_materialize in feast_apply.downstream_list