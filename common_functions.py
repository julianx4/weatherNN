from itertools import groupby
from operator import itemgetter
import datetime
import math
import pandas as pd
import pytz
import pysolar.solar as ps

tz = pytz.timezone('Europe/Berlin')

THRESHOLD_SOLAR_VALUE_SUNSHINE = 0.7

def solar_radiation_threshold_by_time_and_day(timestamp, factor, tz=tz):
    date = datetime.datetime.fromtimestamp(timestamp, tz=tz)
    latitude = 48.4723
    longitude = 9.2117
    altitude_deg = ps.get_altitude(latitude, longitude, date)
    azimuth_deg = ps.get_azimuth(latitude, longitude, date)
    power_output = ps.radiation.get_radiation_direct(date, altitude_deg)
    return float(max(150, power_output * factor))


def solar_radiation_threshold_by_week(timestamp, factor):
    sunny_by_week = [158, 206, 256, 314, 376, 438, 483, 506, 533, 580, 626, 
                    662, 709, 739, 764, 797, 825, 875, 922, 948, 953, 970, 
                    974, 994, 989, 992, 995, 981, 966, 949, 930, 911, 903, 
                    882, 872, 844, 796, 764, 723, 678, 637, 597, 533, 487, 
                    430, 383, 352, 320, 289, 252, 216, 175, 145]
    
    date = datetime.datetime.fromtimestamp(timestamp)
    dt_local = date.astimezone(tz)
    week_number = dt_local.isocalendar()[1]
    return sunny_by_week[week_number-1] * factor
    
def convert_to_normalized_weeknumber(timestamp):
    date = datetime.datetime.fromtimestamp(timestamp)
    dt_local = date.astimezone(tz)
    week_number = dt_local.isocalendar()[1]
    week_number_norm = week_number / 53
    return week_number_norm

def convert_to_normalized_hour(timestamp):
    date = datetime.datetime.fromtimestamp(timestamp)
    dt_local = date.astimezone(tz)
    hour = dt_local.hour
    hour_norm = hour / 23
    return hour_norm

def read_csv(file):
    data = pd.read_csv(file)
    for index, row in data.iterrows():
        yield row['time'] / 1_000_000_000, row['tags'], row['min'], row['max'], row['mean'] 


def group_columns(file):
    for timestamp, values in groupby(read_csv(file), key=itemgetter(0)):
        d = {}
        for (_, topic, value_min, value_max, value_mean) in values:
            if not math.isnan(value_min) and not math.isnan(value_max) and not math.isnan(value_mean):
                d[topic] = {'min': value_min, 'max': value_max, 'mean': value_mean}
        
        try:
            yield (
                (0,1)[d["topic=Pfullingen/solarradiation"]['max'] > solar_radiation_threshold_by_time_and_day(timestamp, THRESHOLD_SOLAR_VALUE_SUNSHINE)],
                (d["topic=Pfullingen/temperature"]['min'] - 20) / 35, 
                (d["topic=Pfullingen/temperature"]['max'] - 20) / 35, 
                (0,1)[d["topic=Pfullingen/temperature"]['min'] < 0],
                (0,1)[d["topic=Pfullingen/windgust"]['max'] > 20],
                (0,1)[d["topic=Pfullingen/hourlyrain"]['mean'] > 0],
                (d["topic=Pfullingen/temperature"]['mean'] - 20) / 35, 
                (d["topic=Pfullingen/windspeed"]['mean']- 5) / 10,
                (d["topic=Pfullingen/hourlyrain"]['mean']) / 10, 
                (d["topic=Pfullingen/pressure"]['mean'] - 500) / 1000,
                (d["topic=Pfullingen/dewpoint"]['mean'] - 20) / 30,
                (d["topic=Pfullingen/winddir"]['mean'] - 180) / 360,
                (d["topic=Pfullingen/humidity"]['mean'] - 50) / 100,
                (d["topic=Pfullingen/solarradiation"]['max'] - 500) / 1000,
                (d["topic=Pfullingen/solarradiation"]['mean'] - 500) / 1000, 
                convert_to_normalized_weeknumber(timestamp),
                convert_to_normalized_hour(timestamp)
            )
        except KeyError:
            pass

