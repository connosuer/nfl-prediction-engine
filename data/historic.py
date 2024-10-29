# Data Cleaning for historic.csv

# imports
import pandas as pd

df = pd.read_csv('data/historic.csv')

"""
CLEANING
- Remove data prior to 2010
- Remove Null Columns
- Remove weather humidity and detail (not enough data)
"""

df = df[df['schedule_season'] >= 2010]
df = df.drop(columns=['Unnamed: 17', 'Unnamed: 18', 
                    'Unnamed: 19', 'Unnamed: 20', 
                    'Unnamed: 21', 'Unnamed: 22',
                    'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25'])
print(df.info())

df.to_csv('data/historic_clean.csv', index=False)