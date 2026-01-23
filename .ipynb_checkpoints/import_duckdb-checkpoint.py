import duckdb

con = duckdb.connect("prescribing.duckdb")
print(con.execute("SHOW TABLES").fetchall())

df = con.execute("SELECT * FROM some_table").df()
print(df.head())
