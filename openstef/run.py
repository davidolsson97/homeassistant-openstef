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

import pandas as pd

print("Hämtar historiska 15-minutersvärden...", flush=True)

result = client.query("""
    SELECT MEAN("value") AS "load"
    FROM "kW"
    WHERE "entity_id" = 'lastbalanserare_importerad_effekt'
      AND time > now() - 8d
    GROUP BY time(15m) fill(none)
""")

df = pd.DataFrame(list(result.get_points()))

print("Antal 15-minutersvärden:", len(df), flush=True)

if not df.empty:
    print("Första tidpunkt:", df["time"].iloc[0], flush=True)
    print("Sista tidpunkt:", df["time"].iloc[-1], flush=True)

if not df.empty:
    tider = pd.to_datetime(df["time"], utc=True).sort_values()

    print("Första fem tidsstämplarna:", flush=True)
    for tid in tider.head(5):
        print(tid, flush=True)

    print("Antal luckor större än 15 minuter:",
          int((tider.diff() > pd.Timedelta(minutes=15)).sum()),
          flush=True)

    print("Minuter i första tidsstämpeln:",
          tider.iloc[0].minute,
          flush=True)

while True:
    time.sleep(60)
