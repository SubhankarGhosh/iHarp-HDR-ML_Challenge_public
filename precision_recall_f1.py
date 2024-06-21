import sys
from os.path import dirname, join as pjoin
import scipy.io as sio
from datetime import datetime, timedelta
from time_series_visualization import *
import pandas as pd
import csv
from sklearn.metrics import (
    mean_squared_error, precision_score, recall_score, f1_score,
    average_precision_score, ndcg_score, balanced_accuracy_score, matthews_corrcoef, fbeta_score, confusion_matrix
)

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

# Ensure 'sltg' and 'anomaly' columns are numeric
original_df['sltg'] = pd.to_numeric(original_df['sltg'], errors='coerce')
generated_df['sltg'] = pd.to_numeric(generated_df['sltg'], errors='coerce')
original_df['anomaly'] = pd.to_numeric(original_df['anomaly'], errors='coerce')
generated_df['anomaly'] = pd.to_numeric(generated_df['anomaly'], errors='coerce')

# Calculate the RMSE between the 'sltg' values of the two DataFrames
rmse = np.sqrt(mean_squared_error(original_df['sltg'], generated_df['sltg']))
print("RMSE:", rmse)

# Calculate precision, recall, and f1 score based on the 'anomaly' column
precision = precision_score(original_df['anomaly'], generated_df['anomaly'])
recall = recall_score(original_df['anomaly'], generated_df['anomaly'])
f1 = f1_score(original_df['anomaly'], generated_df['anomaly'])

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Calculate Area Under the Precision-Recall Curve (AUPRC)
auprc = average_precision_score(original_df['anomaly'], generated_df['anomaly'])
print("AUPRC:", auprc)

# Calculate Normalized Discounted Cumulative Gain (NDCG)
ndcg = ndcg_score([original_df['anomaly']], [generated_df['anomaly']])
print("NDCG:", ndcg)

# Calculate Balanced Accuracy
balanced_acc = balanced_accuracy_score(original_df['anomaly'], generated_df['anomaly'])
print("Balanced Accuracy:", balanced_acc)

# Calculate Matthews Correlation Coefficient (MCC)
mcc = matthews_corrcoef(original_df['anomaly'], generated_df['anomaly'])
print("MCC:", mcc)

# Calculate F-beta Score (where beta > 1 means recall is weighted more)
f2 = fbeta_score(original_df['anomaly'], generated_df['anomaly'], beta=2)
print("F2 Score:", f2)

# Calculate the number of false negatives
conf_matrix = confusion_matrix(original_df['anomaly'], generated_df['anomaly'])
tn, fp, fn, tp = conf_matrix.ravel()
print("False Negatives:", fn)

# Save the results to a log file
log_file_path = os.path.join(script_name, 'Atlantic_City_1962_metrics_log.txt')
with open(log_file_path, 'w') as log_file:
    log_file.write(f"Root Mean Squared Error (RMSE) between 'sltg' values: {rmse}\n")
    log_file.write(f"Precision based on 'anomaly': {precision}\n")
    log_file.write(f"Recall based on 'anomaly': {recall}\n")
    log_file.write(f"F1 Score based on 'anomaly': {f1}\n")
    log_file.write(f"Area Under the Precision-Recall Curve (AUPRC): {auprc}\n")
    log_file.write(f"Normalized Discounted Cumulative Gain (NDCG): {ndcg}\n")
    log_file.write(f"Balanced Accuracy: {balanced_acc}\n")
    log_file.write(f"Matthews Correlation Coefficient (MCC): {mcc}\n")
    log_file.write(f"F2 Score based on 'anomaly': {f2}\n")
    log_file.write(f"False Negatives: {fn}\n")

print("Metrics saved to log file successfully.")