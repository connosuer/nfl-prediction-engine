# Data Cleaning for historic.csv

# imports
import pandas as pd

df = pd.read_csv('historic.csv')

"""
CLEANING
- Remove data prior to 2010
- Remove Null Columns
- Remove weather humidity and detail (not enough data)
- Remove all rows where spread_favorite is missing
- Fill null values
"""

df = df[df['schedule_season'] >= 2010]

df = df.drop(columns=['Unnamed: 17', 'Unnamed: 18', 
                    'Unnamed: 19', 'Unnamed: 20', 
                    'Unnamed: 21', 'Unnamed: 22',
                    'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25',
                    'weather_humidity', 'weather_detail'])

df = df.dropna(subset=['spread_favorite'])


# Features with null: weather_temperature, weather_wind_temperature
# fill nulls with the average of each column grouped by stadium and week
weather = ['weather_temperature', 'weather_wind_mph']
weather_means = df.groupby(['stadium', 'schedule_week'])[weather].transform('mean')

df['weather_temperature'] = df['weather_temperature'].fillna(weather_means['weather_temperature'])
df['weather_wind_mph'] = df['weather_wind_mph'].fillna(weather_means['weather_wind_mph'])

df = df.dropna(subset=['weather_temperature'])
df = df.dropna(subset=['weather_wind_mph'])

df.drop(columns=['stadium', 'stadium_neutral'], inplace=True)

print(df.info())
df.to_csv('historic_clean.csv', index=False)

