
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
from influxdb import InfluxDBClient
from openstef_core.datasets import ForecastInputDataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
log = logging.getLogger("openstef")

MODEL_PATH = Path("/homeassistant/openstef/openstef_modell.pkl")
OPTIONS_PATH = Path("/data/options.json")

MEASUREMENT = "kW"
ENTITY_ID = "lastbalanserare_importerad_effekt"
FORECAST_MEASUREMENT = "openstef_prognos"
MQTT_TOPIC = "openstef/prognos"
DISCOVERY_TOPIC = "homeassistant/sensor/openstef_prognos/config"


def load_options():
    with OPTIONS_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def connect_influx(options):
    client = InfluxDBClient(
        host=options["influxdb_host"],
        port=int(options["influxdb_port"]),
        username=options["influxdb_username"],
        password=options["influxdb_password"],
        database=options["influxdb_database"],
        ssl=options["influxdb_ssl"],
        timeout=30,
    )

    client.ping()
    log.info("Anslutning till InfluxDB fungerar")
    return client


def fetch_history(client, target):
    # Target är början på nästa kvart.
    # Det senaste avslutade kvartet börjar target - 30 minuter.
    last_complete = target - pd.Timedelta(minutes=30)
    first_needed = target - pd.Timedelta(minutes=1455)

    # Extra marginal för att kunna interpolera korta luckor.
    query_start = first_needed - pd.Timedelta(hours=2)
    start_iso = query_start.strftime("%Y-%m-%dT%H:%M:%SZ")

    query = f"""
        SELECT MEAN("value") AS "load"
        FROM "{MEASUREMENT}"
        WHERE "entity_id" = '{ENTITY_ID}'
          AND time >= '{start_iso}'
        GROUP BY time(15m) fill(none)
    """

    result = client.query(query)
    rows = list(result.get_points())

    if not rows:
        raise ValueError("Ingen historik hittades i InfluxDB")

    history = pd.DataFrame(rows)
    history["time"] = pd.to_datetime(history["time"], utc=True)
    history["load"] = pd.to_numeric(history["load"], errors="coerce")

    history = (
        history.set_index("time")
        .sort_index()
        .loc[:last_complete, ["load"]]
    )

    if history.empty:
        raise ValueError("Inga avslutade kvartstimmar hittades")

    full_index = pd.date_range(
        start=query_start.floor("15min"),
        end=last_complete,
        freq="15min",
        tz="UTC",
    )

    history = history.reindex(full_index)

    missing_before = int(history["load"].isna().sum())

    history["load"] = history["load"].interpolate(
        method="time",
        limit=4,
        limit_area="inside",
    )

    missing_after = int(history["load"].isna().sum())

    log.info(
        "Historik: %s kvart, saknade före: %s, efter: %s",
        len(history),
        missing_before,
        missing_after,
    )

    return history


def make_forecast(model, history, target):
    offsets = {
        "load_15min_sedan": 30,
        "load_1h_sedan": 75,
        "load_24h_sedan": 1455,
    }

    features = {}

    for name, minutes in offsets.items():
        timestamp = target - pd.Timedelta(minutes=minutes)

        if timestamp not in history.index:
            raise ValueError(f"Saknad tidpunkt för {name}: {timestamp}")

        value = history.at[timestamp, "load"]

        if pd.isna(value) or not np.isfinite(float(value)):
            raise ValueError(f"Saknat eller ogiltigt värde för {name}")

        features[name] = float(value)

    features["timme"] = target.hour
    features["veckodag"] = target.dayofweek
    features["load"] = np.nan

    future = pd.DataFrame(
        [features],
        index=pd.DatetimeIndex([target], name="time"),
    )

    dataset = ForecastInputDataset(future)
    prediction = model.predict(dataset)

    forecast_kw = float(prediction.data["quantile_P50"].iloc[0])

    if not np.isfinite(forecast_kw):
        raise ValueError("Modellen returnerade en ogiltig prognos")

    return forecast_kw


def publish_mqtt(options, forecast_kw, target):
    username = options.get("mqtt_username", "")
    password = options.get("mqtt_password", "")

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="openstef_ha_green",
    )

    if username:
        client.username_pw_set(username, password)

    client.connect(options["mqtt_host"], int(options["mqtt_port"]), 30)
    client.loop_start()

    try:
        discovery = {
            "name": "OpenSTEF prognos",
            "unique_id": "openstef_prognos",
            "state_topic": MQTT_TOPIC,
            "value_template": "{{ value_json.prognos_kw }}",
            "json_attributes_topic": MQTT_TOPIC,
            "unit_of_measurement": "kW",
            "device_class": "power",
            "state_class": "measurement",
            "icon": "mdi:chart-line",
        }

        info = client.publish(
            DISCOVERY_TOPIC,
            json.dumps(discovery),
            qos=1,
            retain=True,
        )
        info.wait_for_publish(timeout=15)
        if not info.is_published():
            raise RuntimeError("MQTT discovery kunde inte publiceras")

        payload = {
            "prognos_kw": round(forecast_kw, 3),
            "prognos_tid": target.isoformat(),
        }

        info = client.publish(
            MQTT_TOPIC,
            json.dumps(payload),
            qos=1,
            retain=True,
        )
        info.wait_for_publish(timeout=15)
        if not info.is_published():
            raise RuntimeError("MQTT-prognosen kunde inte publiceras")

    finally:
        client.disconnect()
        client.loop_stop()

    log.info("MQTT-prognos publicerad")


def save_influx(client, forecast_kw, target):
    point = {
        "measurement": FORECAST_MEASUREMENT,
        "time": target.isoformat(),
        "fields": {
            "prognos_kw": float(forecast_kw),
        },
    }

    if not client.write_points([point]):
        raise RuntimeError("InfluxDB kunde inte spara prognosen")

    log.info("Prognos sparad i InfluxDB")


def forecast_target():
    now = pd.Timestamp.now(tz="UTC")
    return now.floor("15min") + pd.Timedelta(minutes=15)


def main():
    log.info("OpenSTEF 0.3.0 startar")

    options = load_options()

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Modellfil saknas: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    log.info("Modellen laddad: %s", type(model).__name__)

    last_target = None

    while True:
        target = forecast_target()

        if target != last_target:
            try:
                influx = connect_influx(options)

                try:
                    history = fetch_history(influx, target)
                    forecast_kw = make_forecast(model, history, target)

                    log.info(
                        "Prognos för %s: %.3f kW",
                        target.isoformat(),
                        forecast_kw,
                    )

                    # Publicera först när beräkningen lyckats.
                    publish_mqtt(options, forecast_kw, target)
                    save_influx(influx, forecast_kw, target)

                    last_target = target
                    log.info("Prognoskörning klar")

                finally:
                    influx.close()

            except Exception:
                log.exception("Prognoskörningen misslyckades")

        # Vänta till strax efter nästa kvartsskifte.
        now = pd.Timestamp.now(tz="UTC")
        next_boundary = now.floor("15min") + pd.Timedelta(minutes=15)
        seconds = (next_boundary - now).total_seconds() + 5

        # Vid fel: försök igen efter högst en minut.
        if last_target != target:
            seconds = min(seconds, 60)

        time.sleep(max(1, seconds))


if __name__ == "__main__":
    main()
