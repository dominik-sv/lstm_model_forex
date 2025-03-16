import psycopg2
import logging
from collections import deque

# Establishing the connection
conn = psycopg2.connect(database='postgres')
logging.info("Database connection established.")

# Create a cursor object to interact with the database
cur = conn.cursor()

# Add the ema column to the table if it doesn't exist
cur.execute("""
    ALTER TABLE forex_ticks_seconds ADD COLUMN IF NOT EXISTS ema NUMERIC(10,5);
""")
conn.commit()

cur.execute("""
    SELECT id, normalized_close 
    FROM forex_ticks_seconds
    ORDER BY timestamp;
""")

rows = cur.fetchall()
window = 20
smoothing = 2

# Weights for the previous 20 values, decaying exponentially
weights = [(1 - smoothing / (window + 1)) ** i for i in range(window)]

# Normalize weights so they sum to 1
denominator = sum(weights)
weights = [w / denominator for w in weights]

previous_values = deque(maxlen=window)
emas = []

# Iterate through each row and calculate the EMA using the sliding window of the previous 20 values
for row in rows:
    # Convert 'normalized_close' to float
    previous_values.append(float(row[1]))
    if len(previous_values) == window:
        ema = sum(value * weight for value, weight in zip(previous_values, reversed(weights)))
        emas.append((ema, row[0]))  # row[0] is the 'id'

# Batch update the EMA values in the database
cur.executemany("""
    UPDATE forex_ticks_seconds
    SET ema = %s
    WHERE id = %s;
""", emas)

conn.commit()
logging.info("EMA values updated successfully.")
