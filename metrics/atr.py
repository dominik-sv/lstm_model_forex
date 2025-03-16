import psycopg2
import logging

# Establishing the connection
conn = psycopg2.connect(database='postgres')
logging.info("Database connection established.")

# Create a cursor object to interact with the database
cur = conn.cursor()

# Add the atr column if it doesn't exist
cur.execute("""
    ALTER TABLE forex_ticks_seconds 
    ADD COLUMN IF NOT EXISTS atr NUMERIC(10,5);
""")
conn.commit()

# Step 1: Calculate the ATR and update the atr column for the current group_id
cur.execute("""
    WITH true_range AS (
        SELECT 
            id,
            high - low AS range,
            id_in_group_id
        FROM forex_ticks_seconds
    ),

    atr_calculation AS (
        SELECT 
            id,
            id_in_group_id,
            AVG(range) OVER (ORDER BY id ROWS BETWEEN 12 PRECEDING AND CURRENT ROW) AS atr
        FROM true_range
    ),

    working_zone AS (
        SELECT * 
        FROM atr_calculation
        WHERE id_in_group_id >= 14
    ),

    min_max_atr AS (
        SELECT 
            MIN(atr) AS min_atr_value, 
            MAX(atr) AS max_atr_value
        FROM working_zone
    ),

    atr_normalized_zone AS (
        SELECT 
            wz.id,
            (wz.atr - mm.min_atr_value) / (mm.max_atr_value - mm.min_atr_value) - 0.5 AS atr_normalized
        FROM working_zone wz
        JOIN min_max_atr mm
        ON 1=1
    )

    UPDATE forex_ticks_seconds
    SET atr = anz.atr_normalized
    FROM atr_normalized_zone anz
    WHERE forex_ticks_seconds.id = anz.id;
""")
conn.commit()

# Log the completion of the current group_id
print("ATR calculation and normalization completed.")

# Close the cursor and connection
cur.close()
conn.close()

logging.info("All ATR calculations, normalization, and database updates completed.")
