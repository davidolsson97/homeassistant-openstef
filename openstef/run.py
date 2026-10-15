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

while True:
    time.sleep(60)
