import numpy as np
import pandas as pd
from keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import psycopg2
import logging
from tqdm import tqdm
import json
import random
import time

from tensorflow.keras import mixed_precision  # For mixed precision
mixed_precision.set_global_policy('mixed_float16')  # Enable mixed precision

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Check GPU availability
print(f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")


# parser = argparse.ArgumentParser(description="Train LSTM model with varying parameters.")
# parser.add_argument('--sequence_length', type=int, required=True, help="Sequence length for LSTM")
# parser.add_argument('--batch_size', type=int, required=True, help="Batch size for training")
# parser.add_argument('--epochs', type=int, required=True, help="Number of epochs for training")

# args = parser.parse_args()

# Use the parsed arguments instead of hard-coded values
# sequence_length = args.sequence_length
# batch_size = args.batch_size
# epochs = args.epochs

# Define the possible values for sequence_length, batch_size, and epochs
sequence_length = 50
batch_sizes = 25
epochs_list = 30

metrics = [
    'atr', 'bollinger_bands', 'macd_histogram', 'rsi', 'ema', 
    'up_down2', 'up_down4', 'up_down11', 'up_down61', 
    'roc_2_period', 'roc_4_period', 'roc_11_period', 'roc_61_period', 
    'price_t_minus_1', 'price_t_minus_3', 'price_t_minus_10', 'price_t_minus_60'
]


# Set up logging
start_time = time.time()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Establishing the connection
conn = psycopg2.connect(database='postgres')  # CHANGE ---
logging.info("Database connection established.")

# Create a cursor object to interact with the database
cur = conn.cursor()


metrics_str = ', '.join(metrics)

# Execute the query to retrieve normalized_close, group_id, and output_ema
query = f"""
SELECT normalized_close, group_id, output_ema, {metrics_str}
FROM forex_ticks_seconds
ORDER BY id;
"""
cur.execute(query)
logging.info("SQL query executed successfully.")

# Fetch all results from the executed query
data = cur.fetchall()

# Convert the data into a pandas DataFrame for easier manipulation
df = pd.DataFrame(data, columns=['normalized_close', 'group_id', 'output_ema'] + metrics)

# Convert the normalized_close column from Decimal to float
df['normalized_close'] = df['normalized_close'].astype(float)
df['output_ema'] = df['output_ema'].astype(float)
for metric in metrics:
    df[metric] = df[metric].astype(float)
logging.info(f"Data fetched and converted to DataFrame. Number of rows: {len(df)}")

# Close the cursor and connection
cur.close()
conn.close()
logging.info("Database connection closed.")

# Function to create sequences from a dataset
def create_sequences(df, sequence_length, feature_columns):
    sequences = []
    targets = []
    for i in range(len(df) - sequence_length):
        # Create a sequence containing the specified features (including 'normalized_close' and metrics)
        seq = df[feature_columns].iloc[i:i + sequence_length].values
        sequences.append(seq[:])  # Include the entire sequence
        targets.append(df['output_ema'].iloc[i + sequence_length - 1])  # Target value is the output_ema from the current row
    
    return np.array(sequences), np.array(targets)


# Feature columns including 'normalized_close' and the metrics
feature_columns = ['normalized_close'] + metrics

# Placeholder to store train and test datasets
train_datasets = []
test_datasets = []

# Randomly select 10 unique group_ids for testing
unique_groups = df['group_id'].unique()
test_group_ids = np.random.choice(unique_groups, size=10, replace=False)
logging.info(f"Selected test group IDs: {test_group_ids}")

# Prepare datasets based on group_id
logging.info("Preparing datasets based on group_id.")
for group_id in tqdm(unique_groups, desc="Preparing datasets"):
    group_data = df[df['group_id'] == group_id]
    X, y = create_sequences(group_data, sequence_length, feature_columns)
    
    # Reshape for LSTM input
    X = X.reshape((X.shape[0], X.shape[1], len(feature_columns)))

    if group_id in test_group_ids:
        test_datasets.append((X, y, group_id))
    else:
        train_datasets.append((X, y))

logging.info(f"Total training datasets: {len(train_datasets)}, Total test datasets: {len(test_datasets)}")

# Define a single LSTM model
model = Sequential()
model.add(Input(shape=(sequence_length-1, len(feature_columns))))
model.add(LSTM(200, activation='relu', return_sequences=True))  # Increased units
model.add(LSTM(100, activation='relu'))  # Added second LSTM layer
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
logging.info("LSTM model defined and compiled.")

# Set up the ModelCheckpoint callback to save the model after every epoch
checkpoint_path = "./checkpoints/models/lstm_epoch_{epoch:02d}.keras"
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,   # Save only the model's weights
    save_freq='epoch',        # Save after each epoch
    verbose=1                 # Print a message when saving
)

loss_per_batch = []
logging.info("Starting model training.")
for epoch in range(epochs):
    logging.info(f"Starting epoch {epoch+1}/{epochs}")

    # Randomize the order of datasets before training in each epoch
    random.shuffle(train_datasets)

    for i, (X_train, y_train) in enumerate(train_datasets):
        history = model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=1, callbacks=[checkpoint_callback])
        loss_per_batch.append(history.history['loss'][0])
        logging.info(f"{i+1}/{len(train_datasets)} datasets trained in epoch {epoch+1}")

logging.info("Model training completed on all datasets.")

# Save the final model
model_filename = f"./models/lstm_model_seq{sequence_length}_batch{batch_size}_epoch{epochs}_{metrics}.keras"
model.save(model_filename)
logging.info("Trained model saved")


test_group_ids_list = test_group_ids.tolist()

end_time = time.time()
elapsed_time = end_time - start_time

minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

print(f"Elapsed time: {minutes} minutes and {seconds} seconds")

# Saving model details
model_details = {
    "epochs": epochs,
    "batch_size": batch_size,
    "sequence_length": sequence_length,
    "testing_groups": test_group_ids_list,
    "loss_per_batch": loss_per_batch,
    "time": f'{minutes}:{seconds}'
}

details_filename = f"./json/model_details_seq{sequence_length}_batch{batch_size}_epoch{epochs}.json"
with open(details_filename, 'w') as f:
    json.dump(model_details, f)
