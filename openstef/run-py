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

while True:
    time.sleep(60)
