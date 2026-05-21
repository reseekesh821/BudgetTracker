import pandas as pd
import sqlite3
import os
import uuid
import sys

def ingest_data(csv_file):
    db_file = "data/budget.db"

    # 1. Check if the files exist
    if not os.path.exists(csv_file):
        print(f"Error: Could not find '{csv_file}'. Check your spelling or run the generator script first.")
        return
    
    if not os.path.exists(db_file):
        print(f"Error: Could not find '{db_file}'. Boot up your FastAPI app to create it.")
        return

    # 2. Load the CSV data
    df = pd.read_csv(csv_file)

    # 3. Connect to the SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # 4. Insert each row
    inserted_count = 0
    for index, row in df.iterrows():
        try:
            new_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO transactions (id, date, type, amount, category, note)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (new_id, row['date'], row['type'], row['amount'], row['category'], row['note']))
            inserted_count += 1
        except sqlite3.Error as e:
            print(f"Database error on row {index}: {e}")

    # 5. Save and close
    conn.commit()
    conn.close()

    print(f"Success! {inserted_count} records from '{csv_file}' safely injected.")

if __name__ == "__main__":
    # The Logic Switch
    # If you type a filename in the terminal, it uses that. Otherwise, it defaults to baseline.
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        target_file = "baseline_profile.csv"
        print("No file specified. Defaulting to baseline_profile.csv...")

    ingest_data(target_file)