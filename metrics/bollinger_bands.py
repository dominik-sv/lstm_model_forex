import psycopg2
import logging

# Establishing the connection
conn = psycopg2.connect(database='postgres')
logging.info("Database connection established.")

# Set the multiplier for the Bollinger Bands
multiplier = 2  # You can change this value as needed

# Create a cursor object to interact with the database
cur = conn.cursor()

# Add the atr column if it doesn't exist
cur.execute("""
    ALTER TABLE forex_ticks_seconds 
    ADD COLUMN IF NOT EXISTS bollinger_bands NUMERIC(10,5);
""")
conn.commit()

# Step 1: Calculate Bollinger Bands and update the bollinger_bands column for the current group_id
cur.execute(f"""
    WITH rolling_stats AS (
        SELECT 
            id,
            normalized_close,
            id_in_group_id,
            AVG(normalized_close) OVER (ORDER BY timestamp ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS ma,
            STDDEV(normalized_close) OVER (ORDER BY timestamp ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS std,
            ARRAY_AGG(normalized_close) OVER (ORDER BY timestamp ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS previous_19
        FROM forex_ticks_seconds
    ),
            
    bollinger_calculation AS (
        SELECT 
            id,
            normalized_close,
            ma,
            std,
            previous_19,
            id_in_group_id,
            ma + {multiplier} * std AS upper_band,
            ma - {multiplier} * std AS lower_band,
            CASE 
                WHEN normalized_close BETWEEN ma - {multiplier} * std AND ma + {multiplier} * std THEN 0
                WHEN normalized_close > ma + {multiplier} * std THEN (normalized_close - (ma + {multiplier} * std)) / std
                WHEN normalized_close < ma - {multiplier} * std THEN (normalized_close - (ma - {multiplier} * std)) / std
            END AS bollinger_value
        FROM rolling_stats
    ),
    
    filtered_bollinger AS (
        SELECT bollinger_value AS filtered_bollinger_value, id FROM bollinger_calculation WHERE id_in_group_id >= 20
    ),

    stats AS (
        SELECT 
            MAX(ABS(filtered_bollinger_value)) AS max_abs_value
        FROM filtered_bollinger
    )

    UPDATE forex_ticks_seconds
    SET bollinger_bands = 
        CASE
            WHEN filtered_bollinger_value = 0 THEN 0
            ELSE 0.5 * filtered_bollinger_value / (SELECT max_abs_value FROM stats)
        END
    FROM filtered_bollinger
    WHERE forex_ticks_seconds.id = filtered_bollinger.id;
""")

conn.commit()

# Log the completion of the current group_id
print(f"Bollinger Bands calculation and normalization completed.")

# Close the cursor and connection
cur.close()
conn.close()

logging.info("All Bollinger Bands normalization and database updates completed.")
