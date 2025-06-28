import json
import sqlite3

conn = sqlite3.connect("db/sentences.db")
cursor = conn.cursor()

# get all the rows from the db and info from cols we want
cursor.execute("""
SELECT chinese_traditional, chinese_simplified, english, audio_filename, pinyin
FROM Sentences
""")
rows = cursor.fetchall()

#update static json file
sentences = []
for row in rows:
    trad, simp, english, audio_filename, pinyin_text = row
    sentences.append({
        "chinese_traditional" : trad,
        "chinese_simplified": simp,
        "english": english,
        "audio_filename": audio_filename,
        "pinyin": pinyin_text
    })

with open("static/json/sentences.json", "w", encoding="utf-8") as f:
    json.dump(sentences, f, ensure_ascii=False, indent=2)

conn.close()