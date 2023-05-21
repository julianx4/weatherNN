import meteostat
# Import Meteostat library and dependencies
from datetime import datetime, timedelta
from meteostat import Point, Daily, Hourly
import pandas as pd

start = datetime(2021, 2, 27)
end = datetime.utcnow()
start = end - timedelta(hours=73)

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

