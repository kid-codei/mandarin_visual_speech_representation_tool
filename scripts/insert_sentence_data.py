import sqlite3
import json
import os

# path vars
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sentences_path = os.path.join(base_dir, "data", "sentence_data.json")
db_path = os.path.join(base_dir, "db", "sentences.db")

# Load sentence data
with open(sentences_path, "r", encoding="utf-8") as f:
    sentences = json.load(f)

conn = sqlite3.connect(db_path)
c = conn.cursor()   

# loop through each sample and insert into the database
for sent in sentences:
    c.execute('''
        INSERT INTO Sentences (chinese, english, audio_filename, pitch_json, alignment_json)
        VALUES (?, ?, ?, ?, ?)
    ''', (sent["chinese"], sent["english"], sent["audio_filename"], None, None))  

conn.commit()
conn.close()