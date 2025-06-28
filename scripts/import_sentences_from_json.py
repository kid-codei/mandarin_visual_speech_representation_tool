import json
import sqlite3

with open("data/sentence_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

conn = sqlite3.connect("db/sentences.db")
cursor = conn.cursor()

# create senteneces db schema
cursor.execute("""
CREATE TABLE IF NOT EXISTS Sentences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chinese_traditional TEXT,
    chinese_simplified TEXT,
    english TEXT,
    audio_filename TEXT,
    pinyin TEXT,
    pitch_json TEXT,
    alignment_json TEXT               
)
""")

# insert the rows 
# only have trad, english, and filename for now (original sentences_data.json)
for entry in data:
    cursor.execute("""
        INSERT INTO Sentences (
            chinese_traditional, english, audio_filename     
        ) VALUES (?, ?, ?)
    """, (entry["chinese"], entry["english"], entry["audio_filename"]))

conn.commit()
conn.close()