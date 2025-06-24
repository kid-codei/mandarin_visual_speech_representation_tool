import sqlite3
import os

db_path = "../db/sentences.db"
conn = sqlite3.connect(db_path)
c = conn.cursor()

# print everything
for row in c.execute("SELECT * FROM Sentences"):
    print(row)

conn.close()
