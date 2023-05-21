import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import subprocess
from common_functions import *
from model import *
import meteostat
from datetime import datetime, timedelta
from meteostat import Point, Daily, Hourly
import pandas as pd

command1 = ["influx", "-database", "scwiot", "-execute", "SELECT min(value), max(value), mean(value) FROM weather WHERE time >= now() - 74h GROUP BY time(1h), topic ORDER BY time DESC", "-format", "csv"]
command2 = ["sort", "-k", "3", "-t,"]
command3 = ["sed", "-i", "s/name,tags,time,min,max,mean//", "now_sorted.csv"]
command4 = ["sed", "-i", "1s/^/name,tags,time,min,max,mean\\n/", "now_sorted.csv"]

device = 'cpu'

k = 16  # Dimension of transformer key/query/value vectors 10
heads = 16  # Number of attention heads 2
depth = 12 # Number of transformer blocks 2
num_features = 16  # Number of input features (temperature, pressure)
seq_length = 72  # Length of the sequence

model = Predictor(k, heads, depth, seq_length, num_features, max_seq_len=80).to(device)

#load model:
model.load_state_dict(torch.load('weather_pfullingen.pth', map_location=torch.device(device)))

def load_and_save_meteostat_data():
    end = datetime.utcnow()
    start = end - timedelta(hours=74)

    north_p = Point(52.9911, 9.2159, 400)
    east_p = Point(48.4911, 13.6962, 400)
    south_p = Point(44.4264, 8.91519, 400) #changed a bit because it was in the sea previously 43.9911, 9.2159,
    west_p = Point(48.4911, 4.7356, 400)

    data_north_p = Hourly(north_p, start, end)
    data_north_p = data_north_p.fetch()
    data_east_p = Hourly(east_p, start, end)
    data_east_p = data_east_p.fetch()
    data_south_p = Hourly(south_p, start, end)
    data_south_p = data_south_p.fetch()
    data_west_p = Hourly(west_p, start, end)
    data_west_p = data_west_p.fetch()

    df_data_north_p = pd.DataFrame(data_north_p)
    df_data_east_p = pd.DataFrame(data_east_p)
    df_data_south_p = pd.DataFrame(data_south_p)
    df_data_west_p = pd.DataFrame(data_west_p)

    df_data_north_p.index = pd.to_datetime(df_data_north_p.index)
    df_data_east_p.index = pd.to_datetime(df_data_east_p.index)
    df_data_south_p.index = pd.to_datetime(data_south_p.index)
    df_data_west_p.index = pd.to_datetime(df_data_west_p.index)

    df_data_north_p.index = df_data_north_p.index.view('int64') / 1e9  # Convert from nanoseconds to seconds
    df_data_north_p.index = df_data_north_p.index.astype(int) * 10**9  # Convert from seconds to nanoseconds

    df_data_east_p.index = df_data_east_p.index.view('int64') / 1e9  # Convert from nanoseconds to seconds
    df_data_east_p.index = df_data_east_p.index.astype(int) * 10**9  # Convert from seconds to nanoseconds

    df_data_south_p.index = df_data_south_p.index.view('int64') / 1e9  # Convert from nanoseconds to seconds
    df_data_south_p.index = df_data_south_p.index.astype(int) * 10**9  # Convert from seconds to nanoseconds

    df_data_west_p.index = df_data_west_p.index.view('int64') / 1e9  # Convert from nanoseconds to seconds
    df_data_west_p.index = df_data_west_p.index.astype(int) * 10**9  # Convert from seconds to nanoseconds

    df_data_north_p.index.name = "time"
    df_data_east_p.index.name = "time"
    df_data_south_p.index.name = "time"
    df_data_west_p.index.name = "time"

    df_data_north_p.to_csv('data_north_p_latest.csv', index=True)
    df_data_east_p.to_csv('data_east_p_latest.csv', index=True)
    df_data_south_p.to_csv('data_south_p_latest.csv', index=True)
    df_data_west_p.to_csv('data_west_p_latest.csv', index=True)


def forecast():
    load_and_save_meteostat_data()
    with open("now.csv", "w") as outfile:
        subprocess.run(command1, stdout=outfile)

    with open("now_sorted.csv", "w") as outfile:
        subprocess.run(command2, stdin=open("now.csv"), stdout=outfile)

    subprocess.run(command3)
    subprocess.run(command4)

    file1 = "now_sorted.csv"
    north = 'data_north_p_latest.csv'
    east = 'data_east_p_latest.csv'
    south = 'data_south_p_latest.csv'
    west = 'data_west_p_latest.csv'

    model.eval()
    imported_data_raw = list(group_column(file1, north, east, south, west))
    imported_data = torch.tensor(imported_data_raw)
    eval_data = imported_data[-seq_length:]
    eval_data = eval_data.view(1,seq_length,num_features)

    output = model(eval_data)
    temperature = output[0, :, 0]*35+20
    sol_max =     output[0, :, 3]*1000+500
    sol_mean =    output[0, :, 4]*1000+500
    wind =        output[0, :, 2]*10+5
    rain =        output[0, :, 1]*10

    return temperature, sol_max, sol_mean, wind, rain