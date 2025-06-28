import sqlite3

conn = sqlite3.connect("db/sentences.db")
cursor = conn.cursor()

# make temp table
cursor.execute("ALTER TABLE Sentences RENAME TO TempSentences ")

# make new schema                
cursor.execute("""
    CREATE TABLE Sentences (
        id INTEGER PRIMARY KEY,
        chinese_traditional TEXT,
        chinese_simplified TEXT,
        english TEXT,
        audio_filename TEXT,
        pinyin TEXT,
        pitch_json TEXT,
        alignment_json TEXT
    )
""")

# copy existing data
cursor.execute("""
    INSERT INTO Sentences (
        id, chinese_traditional, english, audio_filename, pitch_json, alignment_json
    )
    SELECT id, chinese, english, audio_filename, pitch_json, alignment_json
    FROM TempSentences
""")

# drop temp table
cursor.execute("DROP TABLE TempSentences")
conn.commit()
conn.close()