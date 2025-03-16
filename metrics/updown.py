import psycopg2
import logging
from tqdm import tqdm
from collections import deque

# Establishing the connection
conn = psycopg2.connect(
    dbname="postgres", 
)

print("Database connection established.")

# Create a cursor object to interact with the database
cur = conn.cursor()

updown_lengths = [2, 4, 11, 61]  # Renamed from `len` to `updown_lengths`

for length in updown_lengths:
    cur.execute(f"""
        ALTER TABLE forex_ticks_seconds ADD COLUMN IF NOT EXISTS up_down{length} NUMERIC(10,5);
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
    up_down_values = []
    previous_normalized_close = None  # To store the value of the previous row

    # Initialize a deque with a maximum length of the current `length`
    previous_values = deque(maxlen=length)
    up_down_values = []

    for row in tqdm(rows):
        tick_id, normalized_close = row
        
        # Add the current value to the deque
        previous_values.append(normalized_close)
        
        # We can only start calculating after we have enough values in the deque
        if len(previous_values) == length:
            # Calculate the jumps between the previous values
            jumps = []
            for i in range(length - 1):  # We compare each of the first pairs (length - 1 comparisons)
                if previous_values[i + 1] > previous_values[i]:
                    jumps.append(0.5)
                elif previous_values[i + 1] < previous_values[i]:
                    jumps.append(-0.5)
                else:
                    jumps.append(0)
            
            # Sum the jumps to get the final up_down score
            total_jumps = sum(jumps)
            # Normalize the total by dividing by `length`
            up_down = total_jumps / (length - 1)
            
            # Append the result
            up_down_values.append((up_down, tick_id))

    # Update the 'up_down' column in the database
    cur.executemany(f"""
        UPDATE forex_ticks_seconds
        SET up_down{length} = %s
        WHERE id = %s;
    """, up_down_values)

    conn.commit()

    print(f'Updown length {length} finished')

# Close the cursor and connection
cur.close()
conn.close()

print("Database update completed.")
