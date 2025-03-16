import numpy as np 
import pandas as pd
import statistics as stat
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import psycopg2
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pyperclip
import json
import argparse
# from automation import metrics  # Import the metrics from the separate module

dataset_plotted_index = 0
plots = 0

metrics = [
    'atr', 'bollinger_bands', 'macd_histogram', 'rsi', 'ema', 
    'up_down2', 'up_down4', 'up_down11', 'up_down61', 
    'roc_2_period', 'roc_4_period', 'roc_11_period', 'roc_61_period', 
    'price_t_minus_1', 'price_t_minus_3', 'price_t_minus_10', 'price_t_minus_60'
]

parser = argparse.ArgumentParser(description="Test LSTM model with provided settings.")
parser.add_argument('--model_name', type=str, required=True, help="Filename of the trained LSTM model.")
parser.add_argument('--details_name', type=str, required=True, help="Filename of the model details JSON file.")

args = parser.parse_args()

# Use the parsed arguments to load the model and details
model_name = args.model_name
details_name = args.details_name

# Load model details from JSON
with open(details_name, 'r') as f:
    model_details = json.load(f)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Establishing the connection
conn = psycopg2.connect(database= 'postgres')           # CHANGE ---
logging.info("Database connection established.")

# Load the pre-trained model with custom_objects
model = load_model(model_name)
logging.info(f"Model loaded from {model_name}")

mega = 1000000
last_x_value = model_details['sequence_length'] - 1

errors = []
jumps = []
errors_nonabs = []
jumps_nonabs = []
error_jump_ratio = []
errors_inc = []
errors_high = []
jumps_inc = []
jumps_high = []
error_high_threshold = 500

# Function to create sequences from a dataset
def create_sequences(df, sequence_length):
    sequences = []
    targets = []
    for i in range(len(df) - sequence_length):
        seq = df[['normalized_close'] + metrics + ['output_ema']].iloc[i:i + sequence_length].values
        sequences.append(seq[:])  # Input sequence
        targets.append(seq[-1]['output_ema'])     # Target value (next normalized close price)
    return np.array(sequences), np.array(targets)


# Create a cursor object to interact with the database
cur = conn.cursor()
price_timeseries = []

for index, group_id in enumerate(model_details['testing_groups']):

    # Execute the query to retrieve normalized_close and the metrics for the specified group_id
    query = f"""
    SELECT normalized_close, {', '.join(metrics)}, output_ema
    FROM forex_ticks_seconds
    WHERE group_id = {group_id}
    ORDER BY id;
    """
    cur.execute(query)
    logging.info("SQL query executed successfully.")

    # Fetch all results from the executed query
    data = cur.fetchall()

    # Convert the data into a pandas DataFrame for easier manipulation
    df = pd.DataFrame(data, columns=['normalized_close'] + metrics + ['output_ema'])

    # Convert all necessary columns to float
    df['normalized_close'] = df['normalized_close'].astype(float)
    df['output_ema'] = df['output_ema'].astype(float)
    for metric in metrics:
        df[metric] = df[metric].astype(float)

    logging.info(f"Data for group {group_id} fetched and converted to DataFrame. Number of rows: {len(df)}")

    # Prepare the data
    X, y = create_sequences(df, model_details['sequence_length'])

    # Reshape for LSTM input
    X = X.reshape((X.shape[0], X.shape[1], len(metrics) + 1))

    # Evaluate the model on the test dataset
    y_pred = model.predict(X)

        # X - every index includes sequence (len 49) of lists of a single number  [a][b][0]
        # y - every index is a single number  [a]
       # y_pred - every index includes a list of a single number  [a][0]

    # Initialize list for storing results
    results = []

    # Initialize counters for each direction
    correct_predictions = 0
    wrong_up = wrong_down = wrong_stay = 0
    total_up = total_down = total_stay = 0

    # Loop through predictions to compare directions
    for i in range(1, len(y_pred)):
        actual_value = y[i-1][0]
        predicted_value = y_pred[i][0]
        
        if y[i-1][0] > y[i-1][0]:
            actual_direction = 'UP'
            total_up += 1
        elif y[i-1][0] < y[i-1][0]:
            actual_direction = 'DOWN'
            total_down += 1
        else:
            actual_direction = 'STAY'
            total_stay += 1
            
        if y_pred[i] > y_pred[i-1]:
            predicted_direction = 'UP'
        elif y_pred[i] < y_pred[i-1]:
            predicted_direction = 'DOWN'
        else:
            predicted_direction = 'STAY'
        
        if actual_direction == predicted_direction:
            correct_predictions += 1
        else:
            if actual_direction == 'UP':
                wrong_up += 1
            elif actual_direction == 'DOWN':
                wrong_down += 1
            elif actual_direction == 'STAY':
                wrong_stay += 1
        
        # Append the result to the list
        results.append([actual_value, actual_direction, predicted_value, predicted_direction])

    # Convert the list to a DataFrame
    results_df = pd.DataFrame(results, columns=['ACTUAL VALUE', 'ACTUAL DIRECTION', 'PREDICTED VALUE', 'PREDICTED DIRECTION'])

    # Save the DataFrame to a CSV file
    # results_df.to_csv('predictions_comparison.csv', index=False, sep=';')

    logging.info("Predictions comparison saved to predictions_comparison.csv")

    # Calculate and log success rate and error statistics
    total_predictions = len(y_pred) - 1  # Since we start from index 1
    success_rate = (correct_predictions / total_predictions) * 100
    wrong_up_percentage = (wrong_up / total_up) * 100 if total_up > 0 else 0
    wrong_down_percentage = (wrong_down / total_down) * 100 if total_down > 0 else 0
    wrong_stay_percentage = (wrong_stay / total_stay) * 100 if total_stay > 0 else 0

    logging.info(f"Prediction Success Rate: {success_rate:.2f}%")
    logging.info(f"Total UP predictions: {total_up}, Wrong UP predictions: {wrong_up} ({wrong_up_percentage:.2f}%)")
    logging.info(f"Total DOWN predictions: {total_down}, Wrong DOWN predictions: {wrong_down} ({wrong_down_percentage:.2f}%)")
    logging.info(f"Total STAY predictions: {total_stay}, Wrong STAY predictions: {wrong_stay} ({wrong_stay_percentage:.2f}%)")

    for i in range(len(y_pred)):
        error = abs(y_pred[i][0]-y[i-1][0]) * mega
        errors.append(error)

    for i in range(len(y_pred)):
        jump = abs(X[i][last_x_value][0] - y[i-1][0]) * mega
        jumps.append(jump)

    for i in range(len(y_pred)):
        error_nonabs = (y_pred[i][0] - y[i-1][0]) * mega
        errors_nonabs.append(error_nonabs)         

    for i in range(len(y_pred)):
        jump_nonabs = (X[i][last_x_value][0] - y[i-1][0]) * mega
        jumps_nonabs.append(jump_nonabs)          
    

    # for i in range(len(y_pred)):
    #     if abs(X[i][last_x_value][0] - y_pred[i][0]) > (X[i][last_x_value][0]-X[i][last_x_value-1][0]):
    #         error_inc = abs(y_pred[i][0]-y[i-1][0]) * mega
    #         errors_inc.append(error_inc)
    #         jump_inc = abs(X[i][last_x_value][0] - y[i-1][0]) * mega
    #         jumps_inc.append(jump_inc)

    # for i in range(len(y_pred)):
    #     if abs(X[i][last_x_value][0] - y[i-1][0]) > (error_high_threshold/mega):
    #         error_high = abs(y_pred[i][0]-y[i-1][0]) * mega
    #         errors_high.append(error_high)
    #         jump_high = abs(X[i][last_x_value][0] - y[i-1][0]) * mega
    #         jumps_high.append(jump_high)


    for i, error in enumerate(errors):
        if jumps[i] != 0:   # Exception for dividing by 0
            errjum_ratio = error / jumps[i]
        error_jump_ratio.append(errjum_ratio)

    # Save variables related to dataset plots
    # if index == dataset_plotted_index:
    #     errors1 = errors[:]
    #     jumps1 = jumps[:]
    #     errors_nonabs1 = errors_nonabs[:]
    #     jumps_nonabs1 = jumps_nonabs[:]
    #     error_jump_ratio1 = error_jump_ratio[:]
    #     price_timeseries = df['normalized_close']
    #     for metric in metrics:
    #         locals()[f"{metric}_timeseries"] = df[metric]

# Close the cursor and connection
cur.close()
conn.close()
logging.info("Database connection closed.")


# MEASURES OF LOCATION
print()
print('MEASURES OF LOCATION')
rounding_digits_1 = 7


error_mean = round(stat.mean(errors), rounding_digits_1)
print(f"Error mean: {error_mean}", end='   ')

jump_mean = round(stat.mean(jumps), rounding_digits_1)
print(f"Jump mean: {jump_mean}")

ej_mean = round(error_mean/jump_mean, rounding_digits_1)

# error_inc_mean = round(stat.mean(errors_inc), rounding_digits_1)
# jump_inc_mean = round(stat.mean(jumps_inc), rounding_digits_1)
# ej_inc_mean = round(error_inc_mean/jump_inc_mean, rounding_digits_1)

# error_high_mean = round(stat.mean(errors_high), rounding_digits_1)
# jump_high_mean = round(stat.mean(jumps_high), rounding_digits_1)
# ej_high_mean = round(error_high_mean/jump_high_mean, rounding_digits_1)

ejr_mean = round(stat.mean(error_jump_ratio), rounding_digits_1)
print(f"Error-Jump Ratio mean: {ejr_mean}")

ejr_05 = 0
ejr_1 = 0
ejr_2 = 0
for ejr in error_jump_ratio:
    if ejr <= 0.5:
        ejr_05 += 1
    if ejr <= 1:
        ejr_1 += 1
    if ejr <= 2:
        ejr_2 += 1
ejr_05p = round(ejr_05/len(error_jump_ratio), rounding_digits_1)
ejr_1p = round(ejr_1/len(error_jump_ratio), rounding_digits_1)
ejr_2p = round(ejr_2/len(error_jump_ratio), rounding_digits_1)



error_med = round(stat.median(errors), rounding_digits_1)
print(f"Error median: {error_med}", end='   ')

jump_med = round(stat.median(jumps), rounding_digits_1)
print(f"Jump median: {jump_med}")

ejr_med = round(stat.median(error_jump_ratio), rounding_digits_1)
print(f"Error-Jump Ratio median: {ejr_med}")


errors_sq = []
for error in errors:
    error_sq = error ** 2
    errors_sq.append(error_sq)
error_mean_sq = round(stat.mean(errors_sq) ** 0.5, rounding_digits_1)
print(f"Square root of mean of squared error: {error_mean_sq}")

print()



# MEASURES OF VARIANCE
print('MEASURES OF VARIANCE')
rounding_digits_2 = rounding_digits_1

error_sd = round(stat.stdev(errors_nonabs), rounding_digits_2)
print(f"Error standard deviation: {error_sd}", end='   ')

jump_sd = round(stat.stdev(jumps_nonabs), rounding_digits_2)
print(f"Jump standard deviation: {jump_sd}")

ejr_sd = round(stat.stdev(error_jump_ratio), rounding_digits_2)
print(f"Error-Jump Ratio standard deviation: {ejr_sd}")


error_Q1 = np.percentile(errors_nonabs, 25)
error_Q3 = np.percentile(errors_nonabs, 75)
error_IQR = round(error_Q3 - error_Q1, rounding_digits_2)
print(f"Error interquartile range: {error_IQR}", end='   ')

jump_Q1 = np.percentile(jumps_nonabs, 25)
jump_Q3 = np.percentile(jumps_nonabs, 75)
jump_IQR = round(jump_Q3 - jump_Q1, rounding_digits_2)
print(f"Jump interquartile range: {jump_IQR}")

ejr_Q1 = np.percentile(error_jump_ratio, 25)
ejr_Q3 = np.percentile(error_jump_ratio, 75)
ejr_IQR = round(ejr_Q3 - ejr_Q1, rounding_digits_2)
print(f"Error-Jump Ratio interquartile range: {ejr_IQR}")

print()



# MEASURES OF SKEWNESS
print('MEASURES OF SKEWNESS')
rounding_digits_3 = rounding_digits_1

error_nonabs_mean = round(stat.mean(errors_nonabs), rounding_digits_3)
print(f"Non-absolute value error mean: {error_nonabs_mean}")

error_nonabs_med = round(stat.median(errors_nonabs), rounding_digits_3)
print(f"Non-absolute value error median: {error_nonabs_med}")



if plots == 1:
    # GRAPHS
    # Timeseries price
    plt.figure(figsize=(10, 6))
    plt.plot(price_timeseries)
    plt.title(f'Dataset {dataset_plotted_index} price timeseries')

    # Timeseries updown
    plt.figure(figsize=(10, 6))
    plt.plot(bollinger_bands_timeseries)
    plt.title(f'Dataset {dataset_plotted_index} bollinger_bands timeseries')

    plt.figure(figsize=(10, 6))
    plt.plot(up_down_timeseries)
    plt.title(f'Dataset {dataset_plotted_index} up_down timeseries')

    plt.figure(figsize=(10, 6))
    plt.plot(ema_timeseries)
    plt.title(f'Dataset {dataset_plotted_index} ema timeseries')

    plt.figure(figsize=(10, 6))
    plt.plot(macd_histogram_timeseries)
    plt.title(f'Dataset {dataset_plotted_index} macd_histogram timeseries')

    plt.figure(figsize=(10, 6))
    plt.plot(rsi_timeseries)
    plt.title(f'Dataset {dataset_plotted_index} rsi timeseries')

    plt.figure(figsize=(10, 6))
    plt.plot(atr_timeseries)
    plt.title(f'Dataset {dataset_plotted_index} atr timeseries')

    # Graph of jumps and errors (abs value)
    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(15, 8), gridspec_kw={'height_ratios': [2, 1]})

    # First plot (original)
    axs[0].plot(errors1, alpha=0.5, label='errors', color="red", linewidth=0.2)
    axs[0].plot(jumps1, alpha=0.5, label='jumps', color="blue", linewidth=0.2)

    axs[0].axhline(y=error_mean, color='red', linestyle='-', label=f'Error Mean ({error_mean})')
    axs[0].axhline(y=error_med, color='red', linestyle='--', label=f'Error Median ({error_med})')
    axs[0].axhline(y=jump_mean, color='blue', linestyle='-', label=f'Jump Mean ({jump_mean})')
    axs[0].axhline(y=jump_med, color='blue', linestyle='--', label=f'Jump Median ({jump_med})')

    axs[0].set_ylim(bottom=0)

    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Change")
    axs[0].legend()
    axs[0].set_title(f"Graph of jumps and errors (abs value) Dataset: {dataset_plotted_index}")

    # Second plot (zoomed in)
    axs[1].plot(errors1, alpha=0.5, label='errors', color="red", linewidth=0.2)
    axs[1].plot(jumps1, alpha=0.5, label='jumps', color="blue", linewidth=0.2)

    axs[1].axhline(y=error_mean, color='red', linestyle='-', label=f'Error Mean ({error_mean})')
    axs[1].axhline(y=error_med, color='red', linestyle='--', label=f'Error Median ({error_med})')
    axs[1].axhline(y=jump_mean, color='blue', linestyle='-', label=f'Jump Mean ({jump_mean})')
    axs[1].axhline(y=jump_med, color='blue', linestyle='--', label=f'Jump Median ({jump_med})')

    # Zoom in the y-axis for the second plot
    axs[1].set_ylim([0, jump_med + jump_sd])

    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Change")

    plt.tight_layout()



    # Error jump ratio
    plt.figure(figsize=(15, 8))

    plt.hist(error_jump_ratio, bins=500, color="purple", label='Error/Jump Ratio')

    plt.axvline(x=ejr_mean, color='green', linestyle='-', label=f'Error/jump ratio Mean ({ejr_mean})')
    plt.axvline(x=ejr_Q1, color='green', linestyle='--', label=f'Error/jump ratio Q1 ({ejr_Q1})')
    plt.axvline(x=ejr_med, color='green', linestyle='--', label=f'Error/jump ratio Median ({ejr_med})')
    plt.axvline(x=ejr_Q3, color='green', linestyle='--', label=f'Error/jump ratio Q3 ({ejr_Q3})')

    plt.xlabel('Error/Jump Ratio')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title(f"Graph of Error/Jump Ratio Dataset: {dataset_plotted_index}")

    plt.tight_layout()
    plt.xlim(0,10)


    # Distribution of errors and jumps
    bins = 500
    histogram_xaxis_len = 2

    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=bins, alpha=0.5, label='errors', color="red")
    plt.hist(jumps, bins=bins, alpha=0.5, label='jumps', color="blue")
    plt.xlim(0, 0.01 * histogram_xaxis_len * mega)
    x_ticks = np.arange(0, 0.01 * histogram_xaxis_len * mega + 1, 1)
    plt.xticks(x_ticks)
    plt.xlabel('Change')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Distribution of errors and jumps for ALL DATASETS')



    # Learning curve
    plt.figure(figsize=(10, 6))
    loss_per_batch = model_details['loss_per_batch']
    batches_per_epoch = int(len(loss_per_batch)/model_details['epochs'])

    # Epochs
    for i in range(model_details['batch_size'], len(loss_per_batch), model_details['batch_size']):
        plt.axvline(x=i, color='red', linestyle='-', linewidth=0.8)
    plt.ylim(0, 0.00001)
    plt.plot(loss_per_batch, label='Loss per Batch')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.title('Loss per Batch during Training (Zoomed)')

    plt.legend()


    plt.show()


# DATAFRAME FOR EXPORT
model_layer_specs = '; '.join(f"{layer.name} - {layer.get_config()['units']}" for layer in model.layers)
results_file = f'./results/model_details_seq{model_details["sequence_length"]}_batch{model_details["batch_size"]}_epoch{model_details["epochs"]}_allMetrics.json'

metrics_json = {
    "model_layer_specs": model_layer_specs,
    "epochs": model_details['epochs'],
    "batch_size": model_details['batch_size'],
    "sequence_length": model_details['sequence_length'],
    "target_column": "Close",
    "error mean/jump mean": ej_mean,        # new
    "error_mean": error_mean,
    # "error_m./jump_m._increase": ej_inc_mean,
    # "error_m./jump_m._high": ej_high_mean,
    "error_mean_sq": error_mean_sq,
    "error_median": error_med,
    "error_sd": error_sd,
    "error_IQR": error_IQR,
    "error_nonabs_mean": error_nonabs_mean,
    "error_nonabs_median": error_nonabs_med,
    "ejr_median": ejr_med,
    "ejr_IQR": ejr_IQR,
    "ejr_per_<0.5": ejr_05p,
    "ejr_per_<1": ejr_1p,
    "ejr_per_<2": ejr_2p,
    "training_time": model_details['time']
}

# Convert the metrics dictionary to a JSON string
metrics_json_str = json.dumps(metrics_json, indent=4)

# Optionally, you can save the JSON string to a file
with open(results_file, 'w') as f:
    f.write(metrics_json_str)

values_to_copy = '\t'.join(str(value) for value in metrics_json.values())
pyperclip.copy(values_to_copy)
