import psycopg2
import logging

# Establishing the connection
conn = psycopg2.connect(database='postgres')
logging.info("Database connection established.")

# Create a cursor object to interact with the database
cur = conn.cursor()

cur.execute("""
    ALTER TABLE forex_ticks_seconds ADD COLUMN IF NOT EXISTS macd_histogram NUMERIC(10,5);
""")
conn.commit()

# Calculate MACD Histogram grouped by group_id
cur.execute(f"""
    WITH ema_12 AS (
        SELECT 
            id,
            id_in_group_id,
            AVG(normalized_close) OVER (ORDER BY timestamp ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) 
            AS ema_12
        FROM forex_ticks_seconds
    ),
    ema_26 AS (
        SELECT 
            id,
            id_in_group_id,
            AVG(normalized_close) OVER (ORDER BY timestamp ROWS BETWEEN 25 PRECEDING AND CURRENT ROW) 
            AS ema_26
        FROM forex_ticks_seconds
    ),
    macd_calculations AS (
        SELECT 
            e12.id,
            e12.id_in_group_id,
            e12.ema_12 - e26.ema_26 AS macd
        FROM ema_12 e12
        JOIN ema_26 e26 ON e12.id = e26.id
    ),
    signal_line_calculations AS (
        SELECT 
            id,
            id_in_group_id,
            AVG(macd) OVER (ORDER BY id ROWS BETWEEN 8 PRECEDING AND CURRENT ROW) 
            AS signal_line
        FROM macd_calculations
    ),
    histogram_calculation AS (
        SELECT 
            mc.id,
            mc.id_in_group_id,
            mc.macd - slc.signal_line AS macd_histogram
        FROM macd_calculations mc
        JOIN signal_line_calculations slc ON mc.id = slc.id
    ),
            
    filtered_macd AS (
        SELECT macd_histogram AS filtered_macd_histogram, id FROM histogram_calculation WHERE id_in_group_id >= 26
    ),

    stats AS (
        SELECT 
            MAX(ABS(filtered_macd_histogram)) AS max_abs_value
        FROM filtered_macd
    )         
   
    UPDATE forex_ticks_seconds
    SET macd_histogram = 
        0.5 * hc.macd_histogram / (SELECT max_abs_value FROM stats)
    FROM histogram_calculation hc
    WHERE forex_ticks_seconds.id = hc.id;
""")

conn.commit()


# Close the cursor and connection
cur.close()
conn.close()

logging.info("All MACD Histogram calculations and database updates by group_id completed.")
