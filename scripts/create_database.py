import sqlite3
import os

# make the db if it doesnt exist yet
# make sure db always made in root dir 
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_dir = os.path.join(base_dir, "db")
os.makedirs(db_dir, exist_ok=True)

# connect to db 
conn = sqlite3.connect(os.path.join(db_dir, "sentences.db"))
c = conn.cursor()

# define the schema
c.execute('''
    CREATE TABLE IF NOT EXISTS Sentences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chinese TEXT,
        english TEXT,
        audio_filename TEXT,
        pitch_json TEXT,
        alignment_json TEXT
    )
''')

conn.commit()
conn.close()
