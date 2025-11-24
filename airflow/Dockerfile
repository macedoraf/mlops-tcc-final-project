FROM apache/airflow:2.9.1-python3.10

USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  build-essential \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*


USER airflow
RUN pip install feast redis pandas pyarrow fastparquet