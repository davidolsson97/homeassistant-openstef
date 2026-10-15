
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

VERSION = "0.4.0"

MODEL_PATH = Path("/homeassistant/openstef/openstef_modell.pkl")
OPTIONS_PATH = Path("/data/options.json")

MEASUREMENT = "kW"
ENTITY_ID = "lastbalanserare_importerad_effekt"
FORECAST_MEASUREMENT = "openstef_prognos"
ERROR_MEASUREMENT = "openstef_prognosfel"

MQTT_TOPIC = "openstef/prognos"
DISCOVERY_TOPIC = "homeassistant/sensor/openstef_prognos/config"

METRICS_TOPIC = "openstef/prognosfel"

METRIC_SENSORS = {
    "mae_24h_kw": {
        "name": "OpenSTEF MAE 24h",
        "unique_id": "openstef_mae_24h",
        "unit": "kW",
        "icon": "mdi:chart-bell-curve",
    },
    "mae_7d_kw": {
        "name": "OpenSTEF MAE 7d",
        "unique_id": "openstef_mae_7d",
        "unit": "kW",
        "icon": "mdi:chart-bell-curve-cumulative",
    },
    "senaste_fel_kw": {
        "name": "OpenSTEF senaste prognosfel",
        "unique_id": "openstef_senaste_prognosfel",
        "unit": "kW",
        "icon": "mdi:delta",
    },
    "antal_24h": {
        "name": "OpenSTEF jämförda kvartstimmar 24h",
        "unique_id": "openstef_antal_24h",
        "unit": None,
        "icon": "mdi:counter",
    },
}


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
    last_complete = target - pd.Timedelta(minutes=30)
    first_needed = target - pd.Timedelta(minutes=1455)

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
    history["load"] = pd.to_numeric(
        history["load"], errors="coerce"
    )

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
            raise ValueError(
                f"Saknad tidpunkt för {name}: {timestamp}"
            )

        value = history.at[timestamp, "load"]

        if pd.isna(value) or not np.isfinite(float(value)):
            raise ValueError(
                f"Saknat eller ogiltigt värde för {name}"
            )

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

    forecast_kw = float(
        prediction.data["quantile_P50"].iloc[0]
    )

    if not np.isfinite(forecast_kw):
        raise ValueError(
            "Modellen returnerade en ogiltig prognos"
        )

    return forecast_kw


def mqtt_connect(options):
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="openstef_ha_green",
    )

    username = options.get("mqtt_username", "")
    password = options.get("mqtt_password", "")

    if username:
        client.username_pw_set(username, password)

    client.connect(
        options["mqtt_host"],
        int(options["mqtt_port"]),
        30,
    )
    client.loop_start()

    return client


def mqtt_publish(client, topic, payload):
    info = client.publish(
        topic,
        json.dumps(payload),
        qos=1,
        retain=True,
    )

    info.wait_for_publish(timeout=15)

    if not info.is_published():
        raise RuntimeError(
            f"MQTT-publicering misslyckades: {topic}"
        )


def publish_forecast(client, forecast_kw, target):
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

    mqtt_publish(client, DISCOVERY_TOPIC, discovery)

    payload = {
        "prognos_kw": round(forecast_kw, 3),
        "prognos_tid": target.isoformat(),
    }

    mqtt_publish(client, MQTT_TOPIC, payload)

    log.info("MQTT-prognos publicerad")


def publish_metrics(client, metrics):
    for key, config in METRIC_SENSORS.items():
        discovery = {
            "name": config["name"],
            "unique_id": config["unique_id"],
            "state_topic": METRICS_TOPIC,
            "value_template": (
                "{{ value_json." + key + " }}"
            ),
            "icon": config["icon"],
        }

        if config["unit"] is not None:
            discovery["unit_of_measurement"] = config["unit"]
            discovery["state_class"] = "measurement"

        mqtt_publish(
            client,
            "homeassistant/sensor/"
            + config["unique_id"]
            + "/config",
            discovery,
        )

    mqtt_publish(client, METRICS_TOPIC, metrics)

    log.info(
        "MAE publicerad: 24h=%s kW, 7d=%s kW, "
        "antal 24h=%s",
        metrics["mae_24h_kw"],
        metrics["mae_7d_kw"],
        metrics["antal_24h"],
    )


def save_forecast(client, forecast_kw, target):
    point = {
        "measurement": FORECAST_MEASUREMENT,
        "time": target.isoformat(),
        "fields": {
            "prognos_kw": float(forecast_kw),
        },
    }

    if not client.write_points([point]):
        raise RuntimeError(
            "InfluxDB kunde inte spara prognosen"
        )

    log.info("Prognos sparad i InfluxDB")


def query_series(client, query, column):
    result = client.query(query)
    rows = list(result.get_points())

    if not rows:
        return pd.Series(dtype="float64")

    frame = pd.DataFrame(rows)

    if column not in frame.columns:
        return pd.Series(dtype="float64")

    frame["time"] = pd.to_datetime(
        frame["time"],
        utc=True,
    )
    frame[column] = pd.to_numeric(
        frame[column],
        errors="coerce",
    )

    series = (
        frame.set_index("time")[column]
        .sort_index()
    )

    return series[
        ~series.index.duplicated(keep="last")
    ]


def calculate_metrics(client):
    now = pd.Timestamp.now(tz="UTC")

    # Den senaste helt avslutade kvartstimmen.
    last_complete = (
        now.floor("15min")
        - pd.Timedelta(minutes=15)
    )

    # Exakt sju dygn inklusive sista kvartstimmen.
    start = (
        last_complete
        - pd.Timedelta(days=7)
        + pd.Timedelta(minutes=15)
    )

    start_iso = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_exclusive = (
        last_complete + pd.Timedelta(minutes=15)
    )
    end_iso = end_exclusive.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    actual_query = f"""
        SELECT MEAN("value") AS "actual_kw"
        FROM "{MEASUREMENT}"
        WHERE "entity_id" = '{ENTITY_ID}'
          AND time >= '{start_iso}'
          AND time < '{end_iso}'
        GROUP BY time(15m) fill(none)
    """

    forecast_query = f"""
        SELECT "prognos_kw"
        FROM "{FORECAST_MEASUREMENT}"
        WHERE time >= '{start_iso}'
          AND time < '{end_iso}'
    """

    actual = query_series(
        client, actual_query, "actual_kw"
    )
    forecast = query_series(
        client, forecast_query, "prognos_kw"
    )

    # Prognoserna sparas på kvartstimmens start.
    # Behåll bara exakta kvartstidsstämplar.
    if not forecast.empty:
        forecast = forecast[
            forecast.index
            == forecast.index.floor("15min")
        ]

    paired = pd.concat(
        [
            actual.rename("actual_kw"),
            forecast.rename("forecast_kw"),
        ],
        axis=1,
        join="inner",
    )

    paired = paired.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    paired = paired[
        (paired.index >= start)
        & (paired.index <= last_complete)
    ]

    if not paired.empty:
        paired["error_kw"] = (
            paired["actual_kw"]
            - paired["forecast_kw"]
        ).abs()

    start_24h = (
        last_complete
        - pd.Timedelta(hours=24)
        + pd.Timedelta(minutes=15)
    )

    paired_24h = paired[
        paired.index >= start_24h
    ]

    def mean_or_none(frame):
        if frame.empty:
            return None
        return round(
            float(frame["error_kw"].mean()),
            3,
        )

    latest_error = None
    latest_time = None

    if not paired.empty:
        latest = paired.iloc[-1]
        latest_error = round(
            float(latest["error_kw"]),
            3,
        )
        latest_time = paired.index[-1].isoformat()

    metrics = {
        "mae_24h_kw": mean_or_none(paired_24h),
        "mae_7d_kw": mean_or_none(paired),
        "senaste_fel_kw": latest_error,
        "antal_24h": int(len(paired_24h)),
        "antal_7d": int(len(paired)),
        "senaste_jamforda_kvart": latest_time,
        "uppdaterad": now.isoformat(),
    }

    return metrics


def save_metrics(client, metrics):
    fields = {
        "antal_24h": int(metrics["antal_24h"]),
        "antal_7d": int(metrics["antal_7d"]),
    }

    for key in (
        "mae_24h_kw",
        "mae_7d_kw",
        "senaste_fel_kw",
    ):
        if metrics[key] is not None:
            fields[key] = float(metrics[key])

    point = {
        "measurement": ERROR_MEASUREMENT,
        "time": metrics["uppdaterad"],
        "fields": fields,
    }

    if not client.write_points([point]):
        raise RuntimeError(
            "InfluxDB kunde inte spara prognosfelen"
        )

    log.info("Prognosstatistik sparad i InfluxDB")


def forecast_target():
    now = pd.Timestamp.now(tz="UTC")
    return (
        now.floor("15min")
        + pd.Timedelta(minutes=15)
    )


def main():
    log.info("OpenSTEF %s startar", VERSION)

    options = load_options()

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Modellfil saknas: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    log.info(
        "Modellen laddad: %s",
        type(model).__name__,
    )

    last_forecast_target = None
    last_metrics_target = None

    while True:
        target = forecast_target()

        need_forecast = (
            target != last_forecast_target
        )
        need_metrics = (
            target != last_metrics_target
        )

        if need_forecast or need_metrics:
            influx = None
            mqtt_client = None

            try:
                influx = connect_influx(options)
                mqtt_client = mqtt_connect(options)

                if need_forecast:
                    try:
                        history = fetch_history(
                            influx, target
                        )
                        forecast_kw = make_forecast(
                            model, history, target
                        )

                        log.info(
                            "Prognos för %s: %.3f kW",
                            target.isoformat(),
                            forecast_kw,
                        )

                        # Spara först, publicera sedan.
                        save_forecast(
                            influx, forecast_kw, target
                        )
                        publish_forecast(
                            mqtt_client,
                            forecast_kw,
                            target,
                        )

                        last_forecast_target = target
                        log.info("Prognoskörning klar")

                    except Exception:
                        log.exception(
                            "Prognoskörningen misslyckades"
                        )

                # MAE beräknas även om prognosen
                # för nästa kvart skulle misslyckas.
                if need_metrics:
                    try:
                        metrics = calculate_metrics(
                            influx
                        )

                        save_metrics(influx, metrics)
                        publish_metrics(
                            mqtt_client, metrics
                        )

                        last_metrics_target = target

                    except Exception:
                        log.exception(
                            "Beräkning av MAE misslyckades"
                        )

            except Exception:
                log.exception(
                    "Anslutning eller körning misslyckades"
                )

            finally:
                if mqtt_client is not None:
                    mqtt_client.disconnect()
                    mqtt_client.loop_stop()

                if influx is not None:
                    influx.close()

        now = pd.Timestamp.now(tz="UTC")
        next_boundary = (
            now.floor("15min")
            + pd.Timedelta(minutes=15)
        )

        seconds = (
            next_boundary - now
        ).total_seconds() + 5

        if (
            last_forecast_target != target
            or last_metrics_target != target
        ):
            seconds = min(seconds, 60)

        time.sleep(max(1, seconds))


if __name__ == "__main__":
    main()
