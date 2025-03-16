# ALTER TABLE forex_ticks_seconds ADD COLUMN IF NOT EXISTS rsi NUMERIC(10,5);
import psycopg2
import logging

# Establishing the connection
conn = psycopg2.connect(database='postgres')
logging.info("Database connection established.")

# Create a cursor object to interact with the database
cur = conn.cursor()

cur.execute("""
    ALTER TABLE forex_ticks_seconds ADD COLUMN IF NOT EXISTS rsi NUMERIC(10,5);
""")
conn.commit()

# Calculate RSI and normalize it between 0 and 1, grouped by group_id and ordered by id
cur.execute(f"""
    WITH price_changes AS (
        SELECT 
            id,
            normalized_close,
            normalized_close - LAG(normalized_close) OVER (ORDER BY id) AS change
        FROM forex_ticks_seconds
    ),
    gains_losses AS (
        SELECT 
            id,
            CASE WHEN change > 0 THEN change ELSE 0 END AS gain,
            CASE WHEN change < 0 THEN ABS(change) ELSE 0 END AS loss
        FROM price_changes
    ),
    avg_gains_losses AS (
        SELECT 
            id,
            AVG(gain) OVER (ORDER BY id ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS avg_gain,
            AVG(loss) OVER (ORDER BY id ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS avg_loss
        FROM gains_losses
    ),
    rsi_calculation AS (
        SELECT 
            id,
        
            avg_gain,
            avg_loss,
            CASE WHEN avg_loss = 0 THEN 100 ELSE 100 - (100 / (1 + (avg_gain / avg_loss))) END AS rsi
        FROM avg_gains_losses
    ),

            
    normalized_rsi AS (
        SELECT 
            id,
            (rsi / 100) - 0.5 AS normalized_rsi
        FROM rsi_calculation
    )
    UPDATE forex_ticks_seconds
    SET rsi = nr.normalized_rsi
    FROM normalized_rsi nr
    WHERE forex_ticks_seconds.id = nr.id
""")

conn.commit()

# Log the completion of the current group_id
print(f"RSI calculation and normalization completed.")

# Close the cursor and connection
cur.close()
conn.close()

logging.info("All RSI calculations and normalization by group_id completed.")
