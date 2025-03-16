import psycopg2
import logging
from tqdm import tqdm

# Establishing the connection
conn = psycopg2.connect(
    dbname="postgres", 
)
print("Database connection established.")

times = [1, 3, 10, 60]  # in seconds

# Create a cursor object to interact with the database
cur = conn.cursor()

for time in tqdm(times):
    # Convert time to minutes as an integer
    time_in_minutes = int(time / 1)

    # Add the new column if it doesn't exist
    cur.execute(f"""
        ALTER TABLE forex_ticks_seconds ADD COLUMN IF NOT EXISTS price_t_minus_{time_in_minutes} NUMERIC(10,5);
    """)
    conn.commit()

    # Fetch the relevant data from the database (not needed for updating, but useful if you want to inspect the values)
    cur.execute(f"""
    WITH prices AS (
        SELECT id, normalized_close 
        FROM forex_ticks_seconds
        ORDER BY timestamp
    )
    SELECT p1.id, 
        p1.normalized_close AS current_price, 
        p2.normalized_close AS price_t_minus_{time_in_minutes}
    FROM prices p1
    LEFT JOIN prices p2 ON p1.id = p2.id + {time}
    ORDER BY p1.id;
    """)
    rows = cur.fetchall()

    # Update the 'price_t_minus_X' column in the database
    cur.execute(f"""
    UPDATE forex_ticks_seconds AS t1
    SET price_t_minus_{time_in_minutes} = t2.normalized_close
    FROM forex_ticks_seconds AS t2
    WHERE t1.id = t2.id + {time};
    """)

    print(f'price_t_minus_{time_in_minutes} updated')

    conn.commit()

# Close the cursor and connection
cur.close()
conn.close()

print("Database update completed.")
