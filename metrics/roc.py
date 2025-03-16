import psycopg2
import logging
from tqdm import tqdm
from collections import deque

# Establishing the connection
conn = psycopg2.connect(
    dbname="postgres",  # Adjust the database name if necessary
)

print("Database connection established.")

# Create a cursor object to interact with the database
cur = conn.cursor()

# List of periods for which we want to calculate RoC (e.g., 1 period, 3 periods, 5 periods, etc.)
roc_periods = [2, 4, 11, 61]


# Loop over each period length
for period in roc_periods:
    # Add new columns for the RoC of each period if they don't exist
    cur.execute(f"""
        ALTER TABLE forex_ticks_seconds ADD COLUMN IF NOT EXISTS roc_{period}_period NUMERIC(10,5);
    """)
    conn.commit()

    # Fetch the relevant data from the database ordered by timestamp
    cur.execute("""
        SELECT id, normalized_close 
        FROM forex_ticks_seconds
        ORDER BY timestamp;
    """)
    rows = cur.fetchall()

    # Initialize variables
    roc_values = []
    previous_values = deque(maxlen=period)  # Create a deque with the size of the current period

    for row in tqdm(rows):
        tick_id, normalized_close = row
        
        # Add the current price to the deque
        previous_values.append(normalized_close)
        
        # We can only start calculating the RoC after we have enough values in the deque
        if len(previous_values) == period:
            # Calculate the RoC using the formula:
            # (current_price - price_n_periods_ago) / price_n_periods_ago * 100
            old_price = previous_values[0]
            roc = ((normalized_close - old_price) / old_price) / 2

            # Append the result (RoC and tick_id)
            roc_values.append((roc, tick_id))

    # Update the 'roc' column in the database for the current period
    cur.executemany(f"""
        UPDATE forex_ticks_seconds
        SET roc_{period}_period = %s
        WHERE id = %s;
    """, roc_values)

    conn.commit()

    print(f'RoC for {period} period(s) finished.')

# Close the cursor and connection
cur.close()
conn.close()

print("Database update completed.")
