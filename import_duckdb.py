import duckdb
import os

DB_PATH = "prescribing.duckdb"
OUT_DIR = "duckdb_table_heads"

os.makedirs(OUT_DIR, exist_ok=True)

con = duckdb.connect(DB_PATH, read_only=True)
tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]

for table in tables:
    try:
        df = con.execute(f"SELECT * FROM {table} LIMIT 5").fetchdf()
        out_path = os.path.join(OUT_DIR, f"{table}_head.csv")
        df.to_csv(out_path, index=False)
        print(f"Exported {out_path}")
    except Exception as e:
        print(f"Failed to export {table}: {e}")

con.close()
