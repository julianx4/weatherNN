import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import torch.nn as nn
from torch.nn import functional as F
from common_functions import *
import numpy as np

file1 = "test.csv"
device = 'cuda'
north = 'data_north_p_latest.csv'
east = 'data_east_p_latest.csv'
south = 'data_south_p_latest.csv'
west = 'data_west_p_latest.csv'
device = 'cuda'

tz = pytz.timezone('Europe/Berlin')

def convert_to_weeknumber_cossin(timestamp):
    date = datetime.datetime.fromtimestamp(timestamp)
    dt_local = date.astimezone(tz)
    week_number = dt_local.isocalendar()[1]
    sin_week = np.sin(2*np.pi*week_number/53).astype(np.float32)
    cos_week = np.cos(2*np.pi*week_number/53).astype(np.float32)
    return sin_week, cos_week

def convert_to_hour_cossin(timestamp):
    date = datetime.datetime.fromtimestamp(timestamp)
    dt_local = date.astimezone(tz)
    hour = dt_local.hour
    sin_hour = np.sin(2*np.pi*hour/23).astype(np.float32)
    cos_hour = np.cos(2*np.pi*hour/23).astype(np.float32)
    return sin_hour, cos_hour


def closest_timestamp(data, timestamp):
    closest_timestamp = min(data.keys(), key=lambda t: abs(t - timestamp))
    return data[closest_timestamp]

def read_csv(file):
    data = pd.read_csv(file)
    for index, row in data.iterrows():
        yield row['time'] / 1_000_000_000, row['tags'], row['min'], row['max'], row['mean'] 

def read_csv_meteostat(file):
    data = pd.read_csv(file)
    timestamp_dict = {}
    for index, row in data.iterrows():
        timestamp = row['time'] / 1_000_000_000
        timestamp_dict[timestamp] = row['pres']                            
    return timestamp_dict

def group_column(file, north, east, south, west):
    meteostat_data_north = read_csv_meteostat(north)
    meteostat_data_east = read_csv_meteostat(east)
    meteostat_data_south = read_csv_meteostat(south)
    meteostat_data_west = read_csv_meteostat(west)
    
    for timestamp, values in groupby(read_csv(file), key=itemgetter(0)):
        pres_value_north = closest_timestamp(meteostat_data_north, timestamp)
        pres_value_east = closest_timestamp(meteostat_data_east, timestamp)
        pres_value_south = closest_timestamp(meteostat_data_south, timestamp)
        pres_value_west = closest_timestamp(meteostat_data_west, timestamp)
        
        d = {}
        for (_, topic, value_min, value_max, value_mean) in values:
            if not math.isnan(value_min) and not math.isnan(value_max) and not math.isnan(value_mean):
                d[topic] = {'min': value_min, 'max': value_max, 'mean': value_mean}
        
        try:
            week_values = convert_to_weeknumber_cossin(timestamp)
            hour_values = convert_to_hour_cossin(timestamp)
            yield (
                (d["topic=Pfullingen/temperature"]['mean'] - 20) / 35, 
                (d["topic=Pfullingen/hourlyrain"]['mean']) / 10, 
                (d["topic=Pfullingen/windspeed"]['mean']- 5) / 10,
                (d["topic=Pfullingen/solarradiation"]['max'] - 500) / 1000,
                (d["topic=Pfullingen/solarradiation"]['mean'] - 500) / 1000, 
                (d["topic=Pfullingen/pressure"]['mean'] - 500) / 1000,
                (d["topic=Pfullingen/winddir"]['mean'] - 180) / 360,
                (d["topic=Pfullingen/humidity"]['mean'] - 50) / 100,
                hour_values[0],
                hour_values[1],
                week_values[0],
                week_values[1],
                float((pres_value_north - 500) / 1000),
                float((pres_value_east - 500) / 1000),
                float((pres_value_south - 500) / 1000),
                float((pres_value_west - 500) / 1000)
                
            )
        except KeyError:
            pass

imported_data_raw = torch.tensor(list(group_column(file1, north, east, south, west)))
#store tensor in file
torch.save(imported_data_raw, 'imported_data_raw.pt')

print(imported_data_raw.shape)