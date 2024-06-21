import sys
from os.path import dirname, join as pjoin
import scipy.io as sio
from datetime import datetime, timedelta
from time_series_visualization import *
import pandas as pd
import csv

# Get the name of the Python file without the extension
script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

# Create a directory with the same name as the Python file
if not os.path.exists(script_name):
    os.makedirs(script_name)

# Read the original CSV file
original_df = pd.read_csv('./filtered_data/Atlantic_City_1950_2005_with_anomalies.csv')

# Ensure 't' column is in datetime format
original_df['t'] = pd.to_datetime(original_df['t'])

# Define the start date and period
start_date = '1962-03-04 00:00:00'
start_datetime = pd.to_datetime(start_date)
end_datetime = start_datetime + pd.Timedelta(weeks=2)

# Create the subset for the specified period from the original DataFrame
original_subset_df = original_df[(original_df['t'] >= start_datetime) & (original_df['t'] < end_datetime)].copy()

# Save the subset to a new CSV file
original_subset_df.to_csv(os.path.join(script_name, 'Atlantic_City_1962_original_subset.csv'), index=False)

# Now create the modified subset with persistence sltg values
# Get the sltg value at the start date
sltg_value_at_start = original_df.loc[original_df['t'] == start_datetime, 'sltg'].values[0]

# Replace all sltg values in the subset with the sltg value at the start date
original_subset_df['sltg'] = sltg_value_at_start

# Update the anomaly column based on the new sltg values
original_subset_df['anomaly'] = (original_subset_df['sltg'] > 3).astype(int)

# Save the modified subset to a new CSV file
original_subset_df.to_csv(os.path.join(script_name, 'Atlantic_City_1962_persistence_subset.csv'), index=False)

print("Original and modified subsets saved to CSV files successfully.")