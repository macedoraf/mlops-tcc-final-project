FROM apache/airflow:2.9.1-python3.10

USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  build-essential \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*


USER airflow
RUN pip install "numpy<2" "protobuf>=5.26.0" "proto-plus>=1.24.0" "google-api-core>=2.18.0" feast redis pandas pyarrow fastparquet