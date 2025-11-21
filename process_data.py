import pandas as pd

working_dir = "Regression"

inside_data = pd.read_csv(f"{working_dir}/garage_temp_history.csv")
inside_data['date'] = pd.to_datetime(inside_data['date']).dt.date
data_temp = inside_data[inside_data['entity_id'] == 'sensor.garage_temp_and_humidity_air_temperature']
daily_avg_temp = data_temp.groupby(data_temp['date'], as_index=False)['state'].mean().round(1)
data_humidity = inside_data[inside_data['entity_id'] == 'sensor.garage_temp_and_humidity_humidity']
daily_avg_humidity = data_humidity.groupby(data_humidity['date'], as_index=False)['state'].mean().round(1)
daily_avg_merged = pd.merge(daily_avg_temp, daily_avg_humidity, on='date', how='inner', suffixes=('_temp', '_humidity'))
daily_avg_merged = daily_avg_merged.rename(columns={'state_temp': 'inside_temp', 'state_humidity': 'inside_humidity'})

outside_data = pd.read_csv(f"{working_dir}/outside_temp_history.csv")
outside_data['date'] = pd.to_datetime(outside_data['date']).dt.date

all_data_merged = pd.merge(outside_data, daily_avg_merged, on='date', how='inner')
all_data_merged = all_data_merged.drop('date', axis=1)
all_data_merged.to_csv(f"{working_dir}/all_data_merged.csv", index=False)