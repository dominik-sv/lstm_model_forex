import subprocess
import itertools
import time

# Define the possible values for sequence_length, batch_size, and epochs
sequence_lengths = [50]
batch_sizes = [25]
epochs_list = [30]

metrics = [
    'atr', 'bollinger_bands', 'macd_histogram', 'rsi', 'ema', 
    'up_down2', 'up_down4', 'up_down11', 'up_down61', 
    'roc_2_period', 'roc_4_period', 'roc_11_period', 'roc_61_period', 
    'price_t_minus_1', 'price_t_minus_3', 'price_t_minus_10', 'price_t_minus_60'
]

# Iterate over all combinations of sequence_length, batch_size, and epochs
for sequence_length, batch_size, epochs in itertools.product(sequence_lengths, batch_sizes, epochs_list):
    # Define the command to run the script with the current parameters
    command = [
        'python3', 'model.py',  # Replace with the actual name of your script
        '--sequence_length', str(sequence_length),
        '--batch_size', str(batch_size),
        '--epochs', str(epochs)
    ]
    
    # Run the script and wait for it to complete
    subprocess.run(command)
    
    time.sleep(5)
    # Name the model file and other outputs based on the current parameters
    model_name = f"./models/lstm_model_seq{sequence_length}_batch{batch_size}_epoch{epochs}.keras"
    details_name = f"./json/model_details_seq{sequence_length}_batch{batch_size}_epoch{epochs}.json"
    
    command2 = [
        'python3', 'test_model.py',  # Replace with the actual name of your script
        '--model_name', model_name,
        '--details_name', details_name
    ]

    subprocess.run(command2)
    time.sleep(10)

    # Log the completion of the current run
    print(f"Finished training model with sequence length {sequence_length}, batch size {batch_size}, and {epochs} epochs.")
