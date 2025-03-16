from concurrent.futures import ProcessPoolExecutor
import psycopg2
import logging
from collections import deque
from tqdm import tqdm

# Function to calculate EMA for a chunk of data
def calculate_ema(chunk, overlap, weights):
    previous_values = deque(overlap, maxlen=window)
    emas = []

    for row in chunk:
        previous_values.append(float(row[1]))
        if len(previous_values) == window:
            ema = sum(value * weight for value, weight in zip(previous_values, reversed(weights)))
            emas.append((ema, row[0] - 300))  # Adjust row[0] as needed
    
    # Return the EMA results and overlap for the next chunk
    return emas, list(previous_values)

# Function to handle the updates in parallel
def update_ema_parallel(chunk, weights, overlap):
    conn = psycopg2.connect(database='postgres')
    cur = conn.cursor()

    # Calculate EMA for the given chunk
    emas, _ = calculate_ema(chunk, overlap, weights)

    # Update the database in batch
    cur.executemany("""
        UPDATE forex_ticks_seconds
        SET output_ema = %s
        WHERE id = %s;
    """, emas)

    conn.commit()
    conn.close()
    return len(emas)  # Return number of updated rows for progress tracking

# Set up logging
logging.info("Database connection established.")

# Fetch all rows
conn = psycopg2.connect(database='postgres')
cur = conn.cursor()

cur.execute("""
    SELECT id, normalized_close 
    FROM forex_ticks_seconds
    ORDER BY timestamp;
""")
rows = cur.fetchall()

window = 300
smoothing = 2

# Precompute weights for the sliding window
weights = [(1 - smoothing / (window + 1)) ** i for i in range(window)]
denominator = sum(weights)
weights = [w / denominator for w in weights]

# Split rows into chunks for parallel processing
chunk_size = len(rows) // 32  # Assuming 32 vCPUs
overlap_size = window - 1  # We need 299 overlapping rows between chunks
chunks = [rows[i:i + chunk_size + overlap_size] for i in range(0, len(rows), chunk_size)]

# Use ProcessPoolExecutor for parallel updates
with ProcessPoolExecutor(max_workers=32) as executor:
    futures = []
    overlap = []

    # Submit each chunk as a separate task
    for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="Processing chunks"):
        future = executor.submit(update_ema_parallel, chunk, weights, overlap)
        futures.append(future)

    # Wait for all tasks to complete and track progress
    total_updates = 0
    for future in tqdm(futures, desc="Updating database in parallel"):
        total_updates += future.result()  # Get the number of rows updated per task

logging.info(f"Successfully updated {total_updates} rows.")
