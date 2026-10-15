import sys
import time

print("OpenSTEF-tillägget startar!", flush=True)
print("Python-version:", sys.version, flush=True)

try:
    from openstef_models.models.forecasting.lgbm_forecaster import LGBMForecaster
    print("OpenSTEF 4.4.8 fungerar!", flush=True)
except Exception as e:
    print("Fel vid import av OpenSTEF:", e, flush=True)
    raise

from pathlib import Path

modellfil = Path("/homeassistant/openstef/openstef_modell.pkl")

if modellfil.is_file():
    print("Modellfilen hittades!", flush=True)
    print("Filstorlek:", modellfil.stat().st_size, "byte", flush=True)
else:
    print("Modellfilen hittades INTE:", modellfil, flush=True)

import joblib

print("Läser in OpenSTEF-modellen...", flush=True)
model = joblib.load(modellfil)
print("Modellen laddades korrekt!", flush=True)
print("Modelltyp:", type(model).__name__, flush=True)

import json
from influxdb import InfluxDBClient

with open("/data/options.json", encoding="utf-8") as f:
    options = json.load(f)

print("Testar anslutningen till InfluxDB...", flush=True)

client = InfluxDBClient(
    host=options["influxdb_host"],
    port=options["influxdb_port"],
    username=options["influxdb_username"],
    password=options["influxdb_password"],
    database=options["influxdb_database"],
    ssl=options["influxdb_ssl"],
    timeout=15,
)

result = client.query(
    'SELECT LAST("value") FROM "kW" '
    "WHERE \"entity_id\" = 'lastbalanserare_importerad_effekt'"
)

print("InfluxDB-anslutningen lyckades!", flush=True)
print("Mätresultat:", list(result.get_points()), flush=True)

while True:
    time.sleep(60)
