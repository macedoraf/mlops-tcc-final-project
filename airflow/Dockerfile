FROM apache/airflow:latest

USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  build-essential \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*


USER airflow
RUN pip3 install feast redis pandas pyarrow fastparquet