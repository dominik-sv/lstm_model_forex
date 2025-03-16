import psycopg2
import logging

# Establishing the connection
conn = psycopg2.connect(database='postgres')
logging.info("Database connection established.")

# Create a cursor object to interact with the database
cur = conn.cursor()

# Execute the query to delete the first row of every group_id
cur.execute("""
    WITH first_rows AS (
        SELECT id
        FROM (
            SELECT 
                id,
                ROW_NUMBER() OVER (PARTITION BY group_id ORDER BY id ASC) AS rn
            FROM forex_ticks
        ) subquery
        WHERE rn = 1
    )
    DELETE FROM forex_ticks
    WHERE id IN (SELECT id FROM first_rows);
""")

# Commit the transaction
conn.commit()

# Close the cursor and connection
cur.close()
conn.close()

logging.info("First row of every group_id deleted.")
