import sys
from os.path import dirname, join as pjoin
import scipy.io as sio
from datetime import datetime, timedelta
from time_series_visualization import *
import pandas as pd
import csv
from sklearn.metrics import mean_squared_error

# Get the name of the Python file without the extension
script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

# Create a directory with the same name as the Python file
if not os.path.exists(script_name):
    os.makedirs(script_name)

# Read the CSV files
original_df = pd.read_csv('./demo_persistence/Atlantic_City_1962_original_subset.csv')
generated_df = pd.read_csv('./demo_persistence/Atlantic_City_1962_persistence_subset.csv')

# Ensure 't' column is in datetime format
original_df['t'] = pd.to_datetime(original_df['t'])
generated_df['t'] = pd.to_datetime(generated_df['t'])

# Ensure the 'sltg' column is numeric
original_df['sltg'] = pd.to_numeric(original_df['sltg'], errors='coerce')
generated_df['sltg'] = pd.to_numeric(generated_df['sltg'], errors='coerce')

# Calculate the RMSE between the 'sltg' values of the two DataFrames
rmse = np.sqrt(mean_squared_error(original_df['sltg'], generated_df['sltg']))
print("RMSE:", rmse)

# Save the RMSE to a log file
log_file_path = os.path.join(script_name, 'Atlantic_City_1962_rmse_log.txt')
with open(log_file_path, 'w') as log_file:
    log_file.write(f"Root Mean Squared Error (RMSE) between 'sltg' values: {rmse}\n")

print("RMSE saved to log file successfully.")