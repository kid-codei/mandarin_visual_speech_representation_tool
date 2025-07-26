import sqlite3
import json

# udpate db with sentences features
# features are pitch and alignment data

# open json
with open("features.json", "r", encoding="utf-8") as f:
    features = json.load(f) 

# connect to sqlite
conn = sqlite3.connect("db/sentences.db")
c = conn.cursor()

# update sentences db with audio pitch and alignment info
for item in features:
    c.execute("""
        UPDATE Sentences
        SET pitch_json = ?, alignment_json = ?
        WHERE audio_filename = ?
    """, (json.dumps(item["pitch"]), json.dumps(item["alignment"]), item["audio_filename"]))

conn.commit()
conn.close()