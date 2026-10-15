ARG BUILD_FROM=python:3.12-slim
FROM ${BUILD_FROM}

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "openstef==4.4.8" \
    lightgbm \
    influxdb \
    paho-mqtt \
    joblib \
    pandas

WORKDIR /app

COPY run.py /app/run.py

CMD ["python", "-u", "/app/run.py"]
