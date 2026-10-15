ARG BUILD_FROM=python:3.12-slim
FROM ${BUILD_FROM}

RUN pip install --no-cache-dir "openstef==4.4.8" influxdb paho-mqtt joblib pandas

WORKDIR /app

COPY run.py /app/run.py

CMD ["python", "-u", "/app/run.py"]
