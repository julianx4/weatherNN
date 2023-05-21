import json
from influxdb import InfluxDBClient
import time
import forecast
from datetime import datetime, timedelta
import sys
run = "test"

client = InfluxDBClient(host='localhost', port=8086)
db_name = 'scwiot'

if len(sys.argv) > 1:
    # The argument is accessed at index 1
    argument = sys.argv[1]
    if argument == "6am_run":
        run = "6am_run"
    elif argument == "12pm_run":
        run = "12pm_run"
    elif argument == "6pm_run":
        run = "6pm_run"
    elif argument == "12am_run":
        run = "12am_run"

temperature, sol_max, sol_mean, wind, rain = forecast.forecast()
forecast_hours = 24
for index, prediction in enumerate(temperature[:forecast_hours]):
    data = [
        {
            "measurement": "iot_forecast",
            "time": (datetime.utcnow() + timedelta(hours=index+1)).strftime('%Y-%m-%dT%H:%M:%SZ'),
            "tags": {
                "topic": "Pfullingen/temperature_forecast_"+run
            },
            "fields": {
                "value": prediction.item(),
                "lookahead": index+1,
            }
        }
    ]
    client.write_points(data, database=db_name)
print("temperature posted")

for index, prediction in enumerate(sol_max[:forecast_hours]):
    data = [
        {
            "measurement": "iot_forecast",
            "time": (datetime.utcnow() + timedelta(hours=index+1)).strftime('%Y-%m-%dT%H:%M:%SZ'),
            "tags": {
                "topic": "Pfullingen/solar_max_forecast_"+run
            },
            "fields": {
                "value": prediction.item(),
                "lookahead": index+1,
            }
        }
    ]
    client.write_points(data, database=db_name)
print("sol_max posted")

for index, prediction in enumerate(sol_mean[:forecast_hours]):
    data = [
        {
            "measurement": "iot_forecast",
            "time": (datetime.utcnow() + timedelta(hours=index+1)).strftime('%Y-%m-%dT%H:%M:%SZ'),
            "tags": {
                "topic": "Pfullingen/solar_mean_forecast_"+run
            },
            "fields": {
                "value": prediction.item(),
                "lookahead": index+1,
            }
        }
    ]
    client.write_points(data, database=db_name)
print("sol_mean posted")

for index, prediction in enumerate(wind[:forecast_hours]):
    data = [
        {
            "measurement": "iot_forecast",
            "time": (datetime.utcnow() + timedelta(hours=index+1)).strftime('%Y-%m-%dT%H:%M:%SZ'),
            "tags": {
                "topic": "Pfullingen/wind_forecast_"+run
            },
            "fields": {
                "value": prediction.item(),
                "lookahead": index+1,
            }
        }
    ]
    client.write_points(data, database=db_name)
print("wind posted")

for index, prediction in enumerate(rain[:forecast_hours]):
    data = [
        {
            "measurement": "iot_forecast",
            "time": (datetime.utcnow() + timedelta(hours=index+1)).strftime('%Y-%m-%dT%H:%M:%SZ'),
            "tags": {
                "topic": "Pfullingen/rain_forecast_"+run
            },
            "fields": {
                "value": prediction.item(),
                "lookahead": index+1,
            }
        }
    ]
    client.write_points(data, database=db_name)
print("rain posted")
